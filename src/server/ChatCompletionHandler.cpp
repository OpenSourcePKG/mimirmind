// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ChatCompletionHandler.hpp"

#include "server/ApiHelpers.hpp"
#include "server/ChatRequestParser.hpp"
#include "server/PromptTrimmer.hpp"
#include "server/RequestDispatcher.hpp"
#include "server/RequestTracker.hpp"
#include "server/SseEncoder.hpp"
#include "server/TenantMetrics.hpp"

#include "model/ResponseCleaner.hpp"
#include "model/ToolCallParser.hpp"
#include "model/ToolCallStreamDetector.hpp"
#include "model/Tokenizer.hpp"
#include "core/log/Log.hpp"
#include "core/security/ScopedTenant.hpp"
#include "runtime/CudaContextPoison.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
#include "runtime/spec/SpeculativeDecoder.hpp"
#include "runtime/thermal/ThermalGuard.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>

namespace mimirmind::server {

using nlohmann::json;

namespace {

/// Drive one request through the ContinuousBatcher, mirroring the callback
/// contract of InferenceEngine::generate() so the existing response
/// formatting can be reused verbatim. Submits the prompt, then delivers
/// each generated token to `onToken` in order; if `onToken` returns false
/// (client gone) the request is cancelled. Returns the full generated
/// stream (including any trailing stop token — the caller strips it, as it
/// does for generate()). M-Cuda.Batch D2e.2.
std::vector<std::int32_t> runViaBatcher(
        runtime::serving::ContinuousBatcher&      batcher,
        std::vector<std::int32_t>                 promptIds,
        const runtime::GenerateParams&            params,
        std::vector<std::int32_t>                 stopIds,
        std::string                               tenantId,
        const std::function<bool(std::int32_t)>&  onToken) {
    auto req = batcher.submit(std::move(promptIds), params.maxNewTokens,
                              std::move(stopIds), std::move(tenantId));
    std::vector<std::int32_t> out;
    std::size_t  next = 0;
    std::int32_t t    = 0;
    bool aborted = false;
    while (req->waitToken(next, t)) {
        out.push_back(t);
        ++next;
        if (onToken && !onToken(t)) { aborted = true; break; }
    }
    if (aborted) {
        batcher.cancel(req);
    }
    if (!req->error.empty()) {
        // Per-tenant quota is checked before the whole-server overload: both
        // set `error`, but a quota rejection is the caller's own fault (429),
        // not server saturation (503).
        if (req->tenantQuotaExceeded) {
            throw runtime::serving::ServingTenantQuotaError(req->error);
        }
        if (req->overloaded) {
            throw runtime::serving::ServingOverloadedError(req->error);
        }
        throw std::runtime_error(req->error);
    }
    return out;
}

} // namespace

ChatCompletionHandler::ChatCompletionHandler(RequestDispatcher&        dispatcher,
                                              RequestTracker&           tracker,
                                              TenantMetrics&            metrics,
                                              const ServerConfig&        cfg)
    : _dispatcher{dispatcher},
      _tracker{tracker},
      _metrics{metrics},
      _cfg{cfg} {}

bool ChatCompletionHandler::prepareChatRequest(
    runtime::InferenceEngine&      targetEngine,
    const ChatRequest&             cr,
    httplib::Response&             res,
    std::vector<std::int32_t>&     promptIds,
    std::vector<std::int32_t>&     stopIds,
    runtime::GenerateParams&       params,
    TrimReport&                    report,
    std::string&                   forcedToolOpener) {
    forcedToolOpener.clear();
    if (cr.messages.empty()) {
        sendError(res, 400, "invalid_request_error",
                  "messages must not be empty");
        return false;
    }

    const auto& tok = targetEngine.tokenizer();
    // The chat-template style is per-model: a multi-model server may serve
    // a gemma4 default alongside a qwen2 "fast" model, and each needs its
    // own template. Resolve it from the REQUESTED model's architecture,
    // not the process-wide default set at construction.
    const model::ChatTemplate::Style style =
        model::ChatTemplate::detectFromArch(
            targetEngine.config().architecture);
    // M-PT: work on a mutable copy so the trim loop can drop entries
    // without touching the parsed request. Also lets us report the
    // original prompt-token count in the response.
    std::vector<model::ChatMessage> msgs = cr.messages;
    promptIds = model::ChatTemplate::encode(
        style, tok, msgs, /*addGenerationPrompt=*/true, cr.tools,
        cr.enableThinking);

    // Debug teacher-forcing: append the raw prefill suffix (no BOS, no
    // special tokens) so the engine prefills the whole sequence in one pass.
    // Empty by default => no-op. See ChatRequest::assistantPrefill.
    if (!cr.assistantPrefill.empty()) {
        const auto suffix = tok.encode(cr.assistantPrefill, /*addBos=*/false);
        promptIds.insert(promptIds.end(), suffix.begin(), suffix.end());
        MM_LOG_INFO("server",
                    "assistant_prefill teacher-forcing: +{} tokens "
                    "(prompt now {})",
                    suffix.size(), promptIds.size());
    }
    // Raw-ID teacher-forcing (exact replay incl. special tokens).
    if (!cr.assistantPrefillIds.empty()) {
        promptIds.insert(promptIds.end(),
                         cr.assistantPrefillIds.begin(),
                         cr.assistantPrefillIds.end());
        MM_LOG_INFO("server",
                    "assistant_prefill_ids teacher-forcing: +{} ids "
                    "(prompt now {})",
                    cr.assistantPrefillIds.size(), promptIds.size());
    }
    // Debug: dump the exact prompt token IDs the engine will prefill, so the
    // same sequence can be replayed through an external oracle (vLLM/HF).
    // Env-gated, no-op in normal serving.
    if (const char* pd = std::getenv("MIMIRMIND_PROMPT_DUMP")) {
        std::ofstream pf(pd, std::ios::trunc);
        for (std::int32_t id : promptIds) {
            pf << id << "\n";
        }
        MM_LOG_INFO("server", "MIMIRMIND_PROMPT_DUMP wrote {} ids to {}",
                    promptIds.size(), pd);
    }

    stopIds = model::ChatTemplate::stopIds(style, tok);
    PromptTrimmer::extendStopIds(tok, cr.stopStrings, stopIds);

    // M-FunctionCalling: when tools are offered, also stop as soon as the model
    // emits one complete native tool call. Gemma 4 otherwise loops the same
    // <|tool_call>…<tool_call|> block until max_tokens (it never emits <turn|>
    // after a call), turning a single tool round into a multi-minute runaway
    // with dozens of duplicate calls. The closing marker stays in the model
    // output — the tool parse below reads the pre-trim `generated` ids.
    if (!cr.tools.empty()) {
        const auto toolStops = model::ChatTemplate::toolCallStopIds(style, tok);
        stopIds.insert(stopIds.end(), toolStops.begin(), toolStops.end());
    }

    params.maxNewTokens = cr.maxTokens > 0 ? cr.maxTokens : _cfg.defaultMaxNew;
    params.stopIds      = stopIds;

    // M-PT — server-side length discipline.
    {
        std::string trimErr;
        if (!PromptTrimmer::applyPromptTrim(msgs, promptIds, params.maxNewTokens,
                             targetEngine.maxContextTokens(),
                             targetEngine.config().contextLength,
                             tok, style, report, trimErr)) {
            sendError(res, 400, "invalid_request_error", trimErr);
            return false;
        }
        if (report.fired()) {
            MM_LOG_INFO("server",
                "prompt-trim: dropped={} orig_tokens={} eff_tokens={} "
                "max_new={}→{} extrapolated_by={}",
                report.droppedMessages, report.originalPromptTokens,
                report.effectivePromptTokens,
                report.maxNewClampedFrom, report.maxNewClampedTo,
                report.contextExtrapolatedBy);
            if (report.contextExtrapolated) {
                MM_LOG_WARN("server",
                    "prompt Tp={} exceeds model-native context {} by {} "
                    "tokens — RoPE extrapolation zone",
                    report.effectivePromptTokens,
                    targetEngine.config().contextLength,
                    report.contextExtrapolatedBy);
            }
        }
        PromptTrimmer::attachTrimHeaders(res, report);
    }

    // M-FunctionCalling Phase 3 — tool_choice:"required". Force a tool call by
    // prefilling the model's native tool-call opener onto the prompt, so the
    // decoder must continue *inside* a call instead of choosing prose (the
    // failure mode small models like Qwen2.5-1.5B hit on some prompts). Done
    // AFTER prompt-trim: a trim re-encodes promptIds from the messages and
    // would otherwise wipe a pre-trim suffix. The opener is a handful of tokens
    // — well within the post-trim maxNew slack. Its text is echoed back through
    // `forcedToolOpener` so handleBlocking can prepend it before parsing (the
    // opener lives in the prompt, not the generated span).
    if (cr.toolChoice == "required" && !cr.tools.empty()) {
        const auto openerIds = model::ChatTemplate::toolCallOpenerIds(style, tok);
        if (!openerIds.empty()) {
            promptIds.insert(promptIds.end(), openerIds.begin(), openerIds.end());
            forcedToolOpener =
                std::string(model::ChatTemplate::toolCallOpenerText(style));
            MM_LOG_INFO("server",
                        "tool_choice=required: prefilled {} tool-call opener "
                        "token(s) (prompt now {})",
                        openerIds.size(), promptIds.size());
        }
    }

    if (cr.hasTemperature) {
        params.sampling.temperature = cr.temperature;
    }
    params.sampling.topP = cr.topP;
    params.sampling.topK = cr.topK;
    params.sampling.seed = cr.seed;

    // Thinking mode must not run greedy. Qwen3 reasoning degenerates into an
    // endless repetition loop under argmax ("No I will not... No I will not...")
    // that no history penalty reliably breaks. When reasoning is enabled and the
    // request is greedy / near-greedy (temp < 0.6 — including the greedy default
    // used for reference-oracle parity), lift sampling to Qwen3's recommended
    // thinking preset (temp 0.6, top_p 0.95, top_k 20) so the trace can escape
    // the loop. This is the "how vLLM does it" guidance for Qwen3 thinking. Only
    // fills values the client left open (topP==1 / topK==0) and never lowers a
    // hotter explicit sampling; non-thinking requests keep the deterministic
    // argmax default untouched (needed for parity tests).
    if (cr.enableThinking.value_or(false) && params.sampling.temperature < 0.6F) {
        params.sampling.temperature = 0.6F;
        if (params.sampling.topP >= 1.0F) params.sampling.topP = 0.95F;
        if (params.sampling.topK == 0)    params.sampling.topK = 20;
        MM_LOG_INFO("server",
                    "thinking sampling floor: reasoning enabled under greedy/near-"
                    "greedy sampling -> temp=0.6 top_p=0.95 top_k=20 "
                    "(Qwen3 anti-loop preset)");
    }

    // M7f — repetition-control penalties.
    //
    // Server-side defaults are opinionated: a 26B-A4B-it Q6_K model was
    // observed falling into a 1300-token repetition loop when no
    // client-side penalty was set, because SamplingParams alone has no
    // history-based mechanism to break out. Applying a mild frequency
    // + repetition penalty by default protects vanilla OpenAI clients
    // (that don't send any penalty) from the same failure mode. Clients
    // that explicitly set a penalty value (including 0) override the
    // default.
    constexpr float        kDefaultFrequencyPenalty  = 0.5F;
    constexpr float        kDefaultRepetitionPenalty = 1.10F;
    constexpr std::uint32_t kDefaultPenaltyWindow    = 64U;

    params.sampling.frequencyPenalty =
        cr.hasFrequencyPenalty  ? cr.frequencyPenalty  : kDefaultFrequencyPenalty;
    params.sampling.presencePenalty =
        cr.hasPresencePenalty   ? cr.presencePenalty   : 0.0F;
    params.sampling.repetitionPenalty =
        cr.hasRepetitionPenalty ? cr.repetitionPenalty : kDefaultRepetitionPenalty;
    params.sampling.penaltyWindow = kDefaultPenaltyWindow;

    // Anti-repeat safety floor. A client that explicitly disables BOTH the
    // frequency and repetition penalties (e.g. frequency_penalty=0 +
    // repetition_penalty=1.0) leaves greedy decoding with no history-based way
    // to escape a degeneration loop — observed via the Pegenaut tool path as a
    // 1500+-token single-phrase repeat ("- Query: ..." / "0649 0649 ..."). Keep
    // a minimal repetition penalty in that fully-unprotected case so the backend
    // never streams garbage, while still honouring any client that sets a real
    // (>1.0 / >0) penalty value.
    // Floor covers two degeneration modes seen via the Pegenaut agentic path:
    //   1. exact-token loop ("0649 0649 ...")     -> repetition penalty breaks it
    //   2. incrementing-template loop             -> the varying number defeats the
    //      ("The text from page 104 ... page 105")   repetition penalty, but the
    //      constant words (text/page/about/...) recur, so a FREQUENCY penalty over
    //      a WIDE window breaks it.
    // Apply both, over a wide window, only when the client left it fully open.
    constexpr float        kAntiLoopFloorRepetition = 1.10F;
    constexpr float        kAntiLoopFloorFrequency  = 0.50F;
    constexpr std::uint32_t kAntiLoopFloorWindow    = 256U;
    if (params.sampling.frequencyPenalty  <= 0.0F &&
        params.sampling.presencePenalty   <= 0.0F &&
        params.sampling.repetitionPenalty <= 1.0F) {
        params.sampling.repetitionPenalty = kAntiLoopFloorRepetition;
        params.sampling.frequencyPenalty  = kAntiLoopFloorFrequency;
        params.sampling.penaltyWindow     = kAntiLoopFloorWindow;
        MM_LOG_INFO("server",
                    "anti-repeat floor: client disabled all penalties; applying "
                    "repetitionPenalty={} frequencyPenalty={} window={} safeguard",
                    kAntiLoopFloorRepetition, kAntiLoopFloorFrequency,
                    kAntiLoopFloorWindow);
    }
    return true;
}

void ChatCompletionHandler::handle(const httplib::Request& req,
                                    httplib::Response&       res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        sendError(res, 400, "invalid_request_error",
                  std::string{"invalid JSON: "} + e.what());
        return;
    }

    ChatRequest cr;
    try {
        cr = parseChatRequest(body);
    } catch (const std::exception& e) {
        sendError(res, 400, "invalid_request_error", e.what());
        return;
    }

    // Thermal admission BEFORE we commit to a stream: a 503 must ship as
    // a plain JSON response, not as half a chunked SSE body. Uses the
    // default engine's thermal guard as a proxy for the whole process —
    // per-engine thermal separation isn't a thing (single iGPU). In M-Munin.3
    // pool mode there is no always-resident default engine (defaultEnginePtr
    // is null); thermal admission is skipped there (MVP — thermal is a
    // single-user/Sleipnir concern, not the pooled multi-model serving path).
    auto* thermalEngine = _dispatcher.defaultEnginePtr();
    if (auto* guard = thermalEngine != nullptr ? thermalEngine->thermalGuard()
                                               : nullptr;
        guard != nullptr) {
        try {
            guard->checkAdmission();
        } catch (const runtime::ThermalLimitExceeded& e) {
            MM_LOG_INFO("server",
                        "thermal refusal: {}", e.what());
            _metrics.recordOverloadRejected(
                core::security::ScopedTenant::active());
            res.set_header("Retry-After", "10");
            sendError(res, 503, "service_unavailable", e.what());
            return;
        }
    }

    // Serving-class load shedding BEFORE we commit to a stream: like the
    // thermal check above, the continuous batcher stands in as a process-wide
    // capacity proxy (the default serving-class engine is the batcher's).
    // Rejecting here ships a clean 503 as plain JSON rather than a truncated
    // SSE body; submit() still enforces the same bound authoritatively for the
    // narrow check-then-submit race.
    if (_cfg.batcher != nullptr && _cfg.batcher->atCapacity()) {
        MM_LOG_WARN("server",
                    "serving overloaded ({}/{} in flight) — shedding with 503",
                    _cfg.batcher->inflight(), _cfg.batcher->maxInflight());
        _metrics.recordOverloadRejected(core::security::ScopedTenant::active());
        res.set_header("Retry-After", "1");
        sendError(res, 503, "service_unavailable",
                  "server overloaded: too many concurrent requests");
        return;
    }

    // Per-tenant fairness pre-check — like the global shed above, ship a clean
    // 429 as plain JSON before committing to an SSE body when the authenticated
    // tenant is already at its concurrent quota (submit() re-checks
    // authoritatively for the narrow check-then-submit race). No-op when
    // per-tenant limiting is off or auth is disabled (empty tenant label).
    if (_cfg.batcher != nullptr) {
        const std::string& tenant = core::security::ScopedTenant::active();
        if (_cfg.batcher->atCapacityForTenant(tenant)) {
            MM_LOG_WARN("server",
                        "tenant '{}' at quota ({}/{} in flight) — shedding "
                        "with 429",
                        tenant, _cfg.batcher->inflightForTenant(tenant),
                        _cfg.batcher->maxInflightPerTenant());
            _metrics.recordQuotaRejected(tenant);
            res.set_header("Retry-After", "1");
            sendError(res, 429, "rate_limit_exceeded",
                      "tenant quota exceeded: too many concurrent requests "
                      "for this API key");
            return;
        }
    }

    if (cr.stream) {
        handleStream(cr, res);
    } else {
        handleBlocking(cr, res);
    }
}

void ChatCompletionHandler::handleBlocking(const ChatRequest& cr,
                                            httplib::Response& res) {
    auto target = _dispatcher.resolveTarget(cr.model, res);
    if (!target) return;
    auto& engine = *target->engine;

    std::vector<std::int32_t> promptIds;
    std::vector<std::int32_t> stopIds;
    runtime::GenerateParams   params;
    TrimReport                trimReport;
    std::string               forcedToolOpener;
    if (!prepareChatRequest(engine, cr, res, promptIds, stopIds, params,
                            trimReport, forcedToolOpener)) {
        return;
    }

    const auto& tok = engine.tokenizer();

    // Owning tenant for this request (empty when auth is off) — feeds the
    // per-tenant usage metrics recorded at each outcome below. Same thread
    // start-to-finish on the blocking path, so the thread_local is stable.
    const std::string tenant = core::security::ScopedTenant::active();

    runtime::GenerateStats stats;
    std::vector<std::int32_t> generated;

    // Reserve the response id up-front so the /v1/system/status
    // snapshot can carry it while the request is still running.
    const std::string respId = makeRequestId();
    _tracker.begin(respId, promptIds.size(),
                   params.maxNewTokens, /*streaming=*/false);
    RequestTracker::Guard requestGuard{&_tracker, respId};

    auto onPrefillProgress =
        [this, &respId](const runtime::InferenceEngine::PrefillProgress& p)
            -> bool {
            _tracker.updatePrefillProgress(respId, p.blocksDone, p.blocksTotal, p.elapsedMs);
            return true;
        };
    auto onPrefillDone =
        [this, &respId](const runtime::InferenceEngine::PrefillDone&) {
            _tracker.markPrefillDone(respId);
        };
    auto onToken = [this, &respId](std::int32_t) -> bool {
        _tracker.incrementDecodeTokens(respId);
        return true;
    };

    // M-Cuda.Batch D2e.2 — when a continuous batcher is wired for the
    // default (serving-class) engine, requests to it are serviced by the
    // batcher's worker thread (multi-tenant continuous batching) instead of
    // the per-engine-mutex single-session generate(). The batcher owns the
    // engine's serving state + GPU stream, so this path must NOT also take
    // the engine mutex or call generate() concurrently.
    const bool useBatcher =
        (_cfg.batcher != nullptr &&
         target->engine == _dispatcher.defaultEnginePtr());
    if (useBatcher) {
        try {
            generated = runViaBatcher(*_cfg.batcher, promptIds, params,
                                      stopIds, tenant, onToken);
        } catch (const runtime::serving::ServingTenantQuotaError& e) {
            // Per-tenant fairness shed, not a bug: retryable 429.
            _metrics.recordQuotaRejected(tenant);
            res.set_header("Retry-After", "1");
            sendError(res, 429, "rate_limit_exceeded", e.what());
            return;
        } catch (const runtime::serving::ServingOverloadedError& e) {
            // Load shedding, not a bug: retryable 503.
            _metrics.recordOverloadRejected(tenant);
            res.set_header("Retry-After", "1");
            sendError(res, 503, "service_unavailable", e.what());
            return;
        } catch (const std::exception& e) {
            MM_LOG_ERROR("server", "batcher generate failed: {}", e.what());
            _metrics.recordError(tenant);
            sendError(res, 500, "server_error",
                      std::string{"generate: "} + e.what());
            return;
        }
    } else {
        // Per-target mutex — requests to different models run in
        // parallel; requests to the same model serialise on its own
        // engine's mutable scratch + sampler state.
        std::lock_guard<std::mutex> lk{*target->mutex};
        try {
            // M9.11.4 — route through the spec-dec orchestrator when a
            // draft is loaded and target->spec is set (only when the
            // resolved engine is the spec-dec target). Bit-identical
            // fall-through to engine.generate() for sampled requests.
            if (target->spec != nullptr) {
                generated = target->spec->generate(
                    promptIds, params, onToken, &stats,
                    onPrefillDone, onPrefillProgress);
            } else {
                generated = engine.generate(promptIds, params,
                                            onToken, &stats,
                                            onPrefillDone,
                                            onPrefillProgress);
            }
        } catch (const runtime::ThermalLimitExceeded& e) {
            MM_LOG_INFO("server",
                        "thermal refusal at engine entry: {}", e.what());
            _metrics.recordOverloadRejected(tenant);
            res.set_header("Retry-After", "10");
            sendError(res, 503, "service_unavailable", e.what());
            return;
        } catch (const std::exception& e) {
            MM_LOG_ERROR("server", "generate failed: {}", e.what());
            // Fail-fast poison guard: an irrecoverable CUDA fault (e.g. a
            // long-context gemma4 request) poisons the shared context, after
            // which EVERY co-resident model (qwen incl.) serves garbage. Hard-
            // exit for a clean supervisor restart instead of limping on a dead
            // context. Recoverable errors fall through to the 500 below.
            runtime::guardCudaPoison("single-session generate()", e.what());
            _metrics.recordError(tenant);
            sendError(res, 500, "server_error",
                      std::string{"generate: "} + e.what());
            return;
        }
    }

    // Debug: dump the raw generated token IDs (pre-cleaning, incl. special
    // tokens) so a follow-up request can teacher-force the exact sequence via
    // assistant_prefill_ids. Env-gated, no-op in normal serving.
    if (const char* td = std::getenv("MIMIRMIND_TOKEN_DUMP")) {
        std::ofstream tf(td, std::ios::trunc);
        for (std::int32_t id : generated) {
            tf << id << "\n";
        }
        MM_LOG_INFO("server", "MIMIRMIND_TOKEN_DUMP wrote {} ids to {}",
                    generated.size(), td);
    }

    // Strip trailing stop tokens from the rendered text so clients don't
    // see "<|im_end|>" tacked onto the answer. Loop because at low
    // temperatures the model could in theory sample several stop tokens
    // in a row before the engine's stop-check breaks the loop.
    std::vector<std::int32_t> visible = generated;
    bool hitStop = stats.hitStop;
    auto isStop = [&](std::int32_t id) {
        if (id == tok.eosId()) return true;
        return std::find(stopIds.begin(), stopIds.end(), id) != stopIds.end();
    };
    while (!visible.empty() && isStop(visible.back())) {
        visible.pop_back();
        hitStop = true;
    }

    const std::string rawText  = tok.decode(visible, /*skipSpecial=*/true);
    // Split thinking out into `reasoning` so it can ride in
    // message.reasoning_content (vLLM / llama.cpp reasoning-model shape)
    // instead of being discarded. preserveThinking keeps the raw thinking
    // inline in content (debug / KV-cache-parity mode).
    std::string reasoning;
    const std::string text     = _cfg.preserveThinking
        ? rawText
        : model::ChatTemplate::cleanResponse(
              model::ChatTemplate::detectFromArch(
                  engine.config().architecture),
              rawText, &reasoning);

    // M-FunctionCalling: when tools were offered and the model emitted a tool
    // call, surface it as structured tool_calls instead of content. Decode
    // WITHOUT skipping specials — Gemma's tool markers are special tokens that
    // skipSpecial=true would strip; Qwen's are plain text (present either way).
    // Dispatch by marker: Gemma's `<|tool_call>` and Qwen's `<tool_call>` are
    // distinct (the leading pipe).
    std::vector<model::ToolCall> toolCalls;
    if (!cr.tools.empty() && cr.toolChoice != "none") {
        // With tool_choice:"required" the native opener was prefilled into the
        // prompt, so it is absent from the generated span — prepend it back so
        // the parser sees a whole tool-call block. Empty in the "auto" case.
        // Decode the pre-trim `generated` (not `visible`): when we stopped on
        // the tool-call closing marker (see toolCallStopIds) that marker is the
        // last token and `visible` has stripped it — the parser needs it to
        // delimit the block.
        const std::string toolText =
            forcedToolOpener + tok.decode(generated, /*skipSpecial=*/false);
        if (model::ToolCallParser::looksLikeGemmaToolCall(toolText)) {
            toolCalls = model::ToolCallParser::parseGemma(toolText);
        } else if (model::ToolCallParser::looksLikeQwenToolCall(toolText)) {
            toolCalls = model::ToolCallParser::parseQwen(toolText);
        } else {
            // Bare-JSON fallback: a reasoning model (Qwen3.6) sometimes emits the
            // call as plain {"name":…,"arguments":…} with no <tool_call> wrapper,
            // which would otherwise leak into content. Run it over the cleaned
            // answer (`text`, already stripped of the <think> reasoning) and gate
            // on the offered tool names so a legit JSON answer is never parsed as
            // a call.
            std::vector<std::string> toolNames;
            toolNames.reserve(cr.tools.size());
            for (const auto& spec : cr.tools) {
                toolNames.push_back(spec.name);
            }
            if (model::ToolCallParser::looksLikeBareJsonToolCall(text, toolNames)) {
                toolCalls = model::ToolCallParser::parseBareJson(text, toolNames);
            }
        }
    }

    const std::int64_t now   = unixNow();
    const std::string finish = !toolCalls.empty() ? "tool_calls"
                             : (hitStop ? "stop" : "length");

    const std::string echoModel = target->id;

    json usage = {
        {"prompt_tokens",     promptIds.size()},
        {"completion_tokens", visible.size()},
        {"total_tokens",      promptIds.size() + visible.size()},
    };
    // Extension over the OpenAI shape: per-request energy delta from
    // the RAPL package counter. Quietly omitted when no power monitor
    // was active for this call.
    if (stats.packageJoules > 0.0) {
        usage["package_joules"] = stats.packageJoules;
    }
    // M-PT — length-discipline metadata. Only present when trim / clamp
    // / extrapolation-warn actually fired.
    PromptTrimmer::attachTrimUsage(usage, trimReport);

    json message = {{"role", "assistant"}};
    // Surface the model's thinking separately (vLLM / llama.cpp shape). Present
    // on both the content answer and a tool-call turn — a reasoning model
    // thinks before it decides to call a tool, and clients show that.
    if (!reasoning.empty()) {
        message["reasoning_content"] = reasoning;
    }
    if (toolCalls.empty()) {
        message["content"] = text;
    } else {
        // OpenAI shape: content null, calls under tool_calls[]. arguments is a
        // JSON *string* (already normalised by the parser).
        message["content"] = nullptr;
        json tcArr = json::array();
        for (const auto& call : toolCalls) {
            tcArr.push_back({
                {"id",   call.id},
                {"type", "function"},
                {"function", {
                    {"name",      call.name},
                    {"arguments", call.argumentsJson},
                }},
            });
        }
        message["tool_calls"] = std::move(tcArr);
    }

    json response = {
        {"id",      respId},
        {"object",  "chat.completion"},
        {"created", now},
        {"model",   echoModel},
        {"choices", json::array({
            json{
                {"index", 0},
                {"message", std::move(message)},
                {"finish_reason", finish},
            },
        })},
        {"usage", std::move(usage)},
    };

    // Spec-dec accept-rate is the headline diagnostic for M9.11.4 — it
    // tells operators whether the draft is earning its keep. Suffix
    // stays empty when spec-dec was disabled or fell through.
    std::string specSuffix;
    if (stats.specDecRounds > 0 && stats.specDecDrafted > 0) {
        const double acceptRate = static_cast<double>(stats.specDecAccepted)
                                / static_cast<double>(stats.specDecDrafted);
        specSuffix = " spec_rounds=" + std::to_string(stats.specDecRounds)
                   + " spec_acc=" + std::to_string(stats.specDecAccepted)
                   + "/" + std::to_string(stats.specDecDrafted)
                   + " spec_rate=" + std::to_string(acceptRate);
    }
    MM_LOG_INFO("server",
                "chat.completion id={} model={} prompt={} cached={} new={} "
                "prefill={:.0f}ms decode={:.0f}ms energy={:.1f}J finish={}{}",
                respId, echoModel,
                promptIds.size(), stats.cachedTokens, visible.size(),
                stats.prefillMs, stats.decodeMs, stats.packageJoules,
                finish, specSuffix);

    _metrics.recordSuccess(tenant, promptIds.size(), visible.size(),
                           stats.packageJoules, stats.prefillMs, stats.decodeMs);
    sendJson(res, 200, response);
}

void ChatCompletionHandler::handleStream(const ChatRequest& cr,
                                          httplib::Response& res) {
    auto target = _dispatcher.resolveTarget(cr.model, res);
    if (!target) return;
    auto& engine = *target->engine;

    std::vector<std::int32_t> promptIds;
    std::vector<std::int32_t> stopIds;
    runtime::GenerateParams   params;
    // M-PT — headers are attached inside prepareChatRequest before the
    // stream body starts, so a streaming client sees the same
    // x-mimirmind-* signals as the blocking path. We don't push the
    // report into a usage chunk here because mimirmind's SSE format
    // doesn't emit a terminal usage chunk (see prefill_done named
    // event for the token-count signal instead).
    TrimReport                trimReport;
    // M-FunctionCalling: the SSE path re-assembles structured tool_calls from
    // the same native markers the blocking path parses (see
    // ToolCallStreamDetector). tool_choice:"required" prefills the opener
    // onto the prompt (Qwen only — see ChatTemplate::toolCallOpenerText); its
    // text seeds the detector so the forced call's body is buffered from the
    // first generated token instead of leaking as content.
    std::string               forcedToolOpener;
    if (!prepareChatRequest(engine, cr, res, promptIds, stopIds, params,
                            trimReport, forcedToolOpener)) {
        return;
    }

    const std::string respId    = makeRequestId();
    const std::int64_t created  = unixNow();
    const std::string echoModel = target->id;

    // Disable proxy buffering / browser caching so SSE chunks reach the
    // client as they are produced.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Accel-Buffering", "no");

    // The provider lambda owns the heavy work — taking the engine
    // mutex, running generate, writing chunks. The HTTP response
    // headers (200 + content-type) go out as soon as cpp-httplib
    // starts invoking the provider, so the client gets immediate
    // feedback that the request was accepted.
    //
    // Shared state lives on the heap so the provider lambda can be
    // re-entered safely if cpp-httplib calls it more than once.
    struct StreamState {
        std::vector<std::int32_t>     promptIds;
        std::vector<std::int32_t>     stopIds;
        runtime::GenerateParams       params;
        std::string                   respId;
        std::int64_t                  created{};
        std::string                   echoModel;
        // Owning tenant, snapshotted on the request thread: the chunked
        // content-provider that submits to the batcher may be invoked on a
        // different pool thread where the thread_local ScopedTenant no longer
        // holds this request's label.
        std::string                   tenantId;
        // Buffers a trailing incomplete UTF-8 codepoint between tokens
        // so SSE deltas always carry valid UTF-8.
        std::string                   utf8Pending;
        // Same UTF-8 hold buffer, for the reasoning_content stream.
        std::string                   reasoningPending;
        // Per-stream filter that swallows the Gemma 4
        // <|channel>thought<channel|> wrapper at the token level,
        // matching the behaviour ChatTemplate::cleanResponse applies in
        // the non-streaming path. No-op for other chat styles.
        model::ResponseCleaner        cleaner;
        // M-FunctionCalling: mirrors the blocking path's marker-dispatch, but
        // token-by-token. toolCallsEnabled gates it off entirely for requests
        // without tools (or tool_choice:"none") so plain-text streaming is
        // byte-for-byte unchanged. style picks which parser
        // (parseQwen/parseGemma) a completed block goes through.
        bool                          toolCallsEnabled{false};
        model::ChatTemplate::Style    style{model::ChatTemplate::Style::QwenChatML};
        model::ToolCallStreamDetector toolCallDetector;
        int                           nextToolCallIndex{0};
        bool                          anyToolCallEmitted{false};
        bool                          done{false};

        StreamState(model::ChatTemplate::Style chatStyle,
                    const model::Tokenizer&    tok,
                    bool                       preserveThinking,
                    bool                       thinkPreClosed)
            // A `<think>` token makes forStyle() start the cleaner INSIDE the
            // think block (thinking-on pre-opens <think>). But when thinking is
            // off — now the DEFAULT, or an explicit enable_thinking=false — the
            // chat template pre-CLOSES the block (`<think>\n\n</think>\n\n`) so
            // generation is answer content from the first token — start the
            // cleaner in content mode (channelStartId = -1), else the whole
            // answer is mislabelled as reasoning_content and delta.content
            // streams empty. thinkPreClosed MUST equal !thinkOn in ChatTemplate.
            : cleaner{(preserveThinking || thinkPreClosed)
                ? model::ResponseCleaner{chatStyle, -1, -1}
                : model::ResponseCleaner::forStyle(chatStyle, tok)},
              style{chatStyle} {}
    };
    const model::ChatTemplate::Style style =
        model::ChatTemplate::detectFromArch(engine.config().architecture);
    auto state = std::make_shared<StreamState>(
        style, engine.tokenizer(), _cfg.preserveThinking,
        // Mirror ChatTemplate's thinkOn = enableThinking.value_or(false): the
        // prompt pre-closes the think block whenever thinking is off (default,
        // or explicit false), so the cleaner must start in content mode.
        /*thinkPreClosed=*/!cr.enableThinking.value_or(false));
    state->promptIds = std::move(promptIds);
    state->stopIds   = std::move(stopIds);
    state->params    = std::move(params);
    state->respId    = respId;
    state->created   = created;
    state->echoModel = echoModel;
    state->tenantId  = core::security::ScopedTenant::active();
    state->toolCallsEnabled = !cr.tools.empty() && cr.toolChoice != "none";
    state->toolCallDetector = model::ToolCallStreamDetector(
        style, state->toolCallsEnabled, forcedToolOpener);

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, state,
         targetEngine = target->engine,
         targetMutex  = target->mutex,
         targetSpec   = target->spec]
        (std::size_t /*offset*/,
         httplib::DataSink& sink) -> bool {
            if (state->done) {
                return false;
            }
            state->done = true;

            auto& targetEng = *targetEngine;
            const auto& tok = targetEng.tokenizer();

            // M-FunctionCalling: parse one completed marker-delimited block
            // (see ToolCallStreamDetector) with the style-appropriate parser
            // and emit it as a `delta.tool_calls` chunk — the SSE shape
            // clients (and Bifröst's Anthropic-SSE translator) expect.
            // Returns false on a write failure (client gone), matching the
            // other Sse write helpers' contract. A block that fails to parse
            // (malformed/truncated) falls back to surfacing its raw text as
            // content rather than silently dropping it.
            auto emitToolCallBlock = [&](const std::string& block) -> bool {
                const std::vector<model::ToolCall> calls =
                    (state->style == model::ChatTemplate::Style::Gemma4)
                        ? model::ToolCallParser::parseGemma(block)
                        : model::ToolCallParser::parseQwen(block);
                if (calls.empty()) {
                    return SseEncoder::writeSseEvent(
                        sink,
                        SseEncoder::buildContentChunk(state->respId, state->created,
                                          state->echoModel, block));
                }
                for (const auto& call : calls) {
                    const std::string callId =
                        "call_" + std::to_string(state->nextToolCallIndex);
                    if (!SseEncoder::writeSseEvent(
                            sink,
                            SseEncoder::buildToolCallChunk(
                                state->respId, state->created, state->echoModel,
                                state->nextToolCallIndex, callId, call.name,
                                call.argumentsJson))) {
                        return false;
                    }
                    ++state->nextToolCallIndex;
                    state->anyToolCallEmitted = true;
                }
                return true;
            };

            // 1. Initial role chunk so clients see {role:"assistant"}.
            if (!SseEncoder::writeSseEvent(
                    sink,
                    SseEncoder::buildRoleChunk(state->respId, state->created,
                                   state->echoModel))) {
                MM_LOG_INFO("server",
                            "stream {}: client closed before role chunk",
                            state->respId);
                return false;
            }

            // 2. Generate under the engine mutex; emit each non-stop
            //    token as a delta-content chunk. The onToken returning
            //    false causes generate() to abort cleanly.
            std::size_t              emittedTokens = 0;
            runtime::GenerateStats   stats;
            bool                     clientGone    = false;

            auto isStop = [&](std::int32_t id) {
                if (id == tok.eosId()) return true;
                for (auto s : state->stopIds) {
                    if (id == s) return true;
                }
                return false;
            };

            auto onToken = [&](std::int32_t id) -> bool {
                if (isStop(id)) {
                    // Do not surface the stop token's text. The engine
                    // checks isStop on the next iteration and exits.
                    // M-FunctionCalling: Gemma 4's tool-call closer is one of
                    // stopIds (see ChatTemplate::toolCallStopIds) — its text
                    // never reaches the detector below, so finalize here
                    // instead when a call was in flight.
                    if (state->toolCallsEnabled && state->toolCallDetector.buffering()) {
                        state->toolCallDetector.closeOnStop();
                        if (!emitToolCallBlock(state->toolCallDetector.completedBlock())) {
                            clientGone = true;
                            return false;
                        }
                        state->toolCallDetector.reset();
                    }
                    return true;
                }
                // Snapshot the per-token progress even for stripped
                // tokens so /v1/system/status reflects real decode work.
                _tracker.incrementDecodeTokens(state->respId);
                std::string txt = tok.decode(
                    std::span<const std::int32_t>{&id, 1},
                    /*skipSpecial=*/true);
                if (txt.empty()) {
                    return true;
                }

                // Split thinking from answer: `txt` becomes the answer content
                // (may be emptied), `reasoning` the thinking fragment. Structural
                // markup (Gemma 4 channel / Qwen3 <think>…</think>) is consumed.
                std::string reasoning;
                const bool hasContent = state->cleaner.feed(id, txt, reasoning);

                // Reasoning delta → delta.reasoning_content (own UTF-8 hold
                // buffer so a multibyte codepoint split across tokens never
                // ships broken). vLLM / llama.cpp reasoning-model shape.
                if (!reasoning.empty()) {
                    state->reasoningPending.append(reasoning);
                    const std::size_t rcut = SseEncoder::utf8IncompleteTailStart(
                        state->reasoningPending);
                    if (rcut > 0) {
                        std::string remit = state->reasoningPending.substr(0, rcut);
                        state->reasoningPending.erase(0, rcut);
                        if (!SseEncoder::writeSseEvent(
                                sink,
                                SseEncoder::buildReasoningChunk(
                                    state->respId, state->created,
                                    state->echoModel, remit))) {
                            clientGone = true;
                            return false;   // abort generate()
                        }
                    }
                }

                if (!hasContent) {
                    return true;
                }

                // M-FunctionCalling: detect a native tool-call marker span
                // within the answer content and buffer it instead of
                // streaming it as text — see ToolCallStreamDetector. `txt`
                // comes back shortened to whatever content (if any) precedes
                // an opener found this token; a no-op when disabled or no
                // marker is in play, so plain-text streaming is unchanged.
                const bool toolCallCompleted =
                    state->toolCallsEnabled && state->toolCallDetector.feed(txt);

                if (!txt.empty()) {
                    state->utf8Pending.append(txt);
                    const std::size_t cut =
                        SseEncoder::utf8IncompleteTailStart(state->utf8Pending);
                    if (cut > 0) {
                        std::string emit = state->utf8Pending.substr(0, cut);
                        state->utf8Pending.erase(0, cut);

                        if (!SseEncoder::writeSseEvent(
                                sink,
                                SseEncoder::buildContentChunk(state->respId, state->created,
                                                  state->echoModel, emit))) {
                            clientGone = true;
                            return false;   // abort generate()
                        }
                        ++emittedTokens;
                    }
                }

                if (toolCallCompleted) {
                    if (!emitToolCallBlock(state->toolCallDetector.completedBlock())) {
                        clientGone = true;
                        return false;
                    }
                    state->toolCallDetector.reset();
                }
                return true;
            };

            // Named SSE event fired between prefill and the first
            // decode token so a streaming client can flip its UX from
            // "reading your prompt" to "answering". The OpenAI stream
            // demuxer ignores named events, browsers pick it up via
            // EventSource.addEventListener.
            auto onPrefillDone =
                [&](const runtime::InferenceEngine::PrefillDone& p) {
                    _tracker.markPrefillDone(state->respId);
                    const json payload = {
                        {"prompt_tokens",    p.promptTokens},
                        {"prefilled_tokens", p.prefilledTokens},
                        {"prefill_ms",       p.prefillMs},
                        {"response_id",      state->respId},
                    };
                    if (!SseEncoder::writeSseNamedEvent(sink, "prefill_done", payload)) {
                        clientGone = true;
                    }
                };

            // Per-block prefill progress, rate-limited so a fast prefill
            // (~10 ms per block on Gemma 4 26B) doesn't fire 34 SSE
            // events in half a second. First and last blocks always
            // emit; in between we throttle to one event per ~200 ms so
            // a browser progress bar updates smoothly on long prompts
            // without spamming short ones.
            double lastProgressMs = -1.0;
            constexpr double kProgressMinIntervalMs = 200.0;
            auto onPrefillProgress =
                [&](const runtime::InferenceEngine::PrefillProgress& p)
                    -> bool {
                    // Snapshot for /v1/system/status polling — kept
                    // outside the SSE-rate-limit branch so the status
                    // block sees every completed transformer layer.
                    _tracker.updatePrefillProgress(state->respId, p.blocksDone, p.blocksTotal, p.elapsedMs);
                    // M7g — poll for a client-side disconnect on every
                    // block. If clientGone flipped between blocks (via
                    // the throttle branch below OR via a broken write
                    // from onToken during a previous prefill call —
                    // rare but possible in edge flows), tell the engine
                    // to stop at the next barrier.
                    if (clientGone) {
                        return false;
                    }
                    const bool isFirst = (p.blocksDone == 1);
                    const bool isLast  = (p.blocksDone == p.blocksTotal);
                    const bool dueByTime =
                        (p.elapsedMs - lastProgressMs) >=
                        kProgressMinIntervalMs;
                    if (!isFirst && !isLast && !dueByTime) {
                        return true;  // no SSE this tick, keep going
                    }
                    lastProgressMs = p.elapsedMs;
                    const json payload = {
                        {"blocks_done",  p.blocksDone},
                        {"blocks_total", p.blocksTotal},
                        {"elapsed_ms",   p.elapsedMs},
                        {"response_id",  state->respId},
                    };
                    if (!SseEncoder::writeSseNamedEvent(sink, "prefill_progress",
                                            payload)) {
                        clientGone = true;
                        return false;  // abort prefill at next barrier
                    }
                    return true;
                };

            std::vector<std::int32_t> generated;
            std::string               errorMessage;
            // Register this request with the status snapshot so
            // /v1/system/status can report prefill+decode progress
            // while it runs. The RAII guard releases on scope exit —
            // covers early error exits and generate() throws.
            _tracker.begin(state->respId,
                           state->promptIds.size(),
                           state->params.maxNewTokens,
                           /*streaming=*/true);
            RequestTracker::Guard requestGuard{&_tracker, state->respId};
            // M-Cuda.Batch D2e.2 — batcher-serviced streaming for the
            // default (serving-class) engine. The batcher drives onToken per
            // decoded token exactly like generate(); it has no prefill-phase
            // callbacks, so a streaming client just sees decode deltas (no
            // prefill_done / prefill_progress events on this path). onToken
            // returning false (client gone) cancels the batcher request.
            const bool useBatcher =
                (this->_cfg.batcher != nullptr &&
                 targetEngine == this->_dispatcher.defaultEnginePtr());
            if (useBatcher) {
                try {
                    generated = runViaBatcher(*this->_cfg.batcher,
                                              state->promptIds, state->params,
                                              state->stopIds, state->tenantId,
                                              onToken);
                } catch (const runtime::serving::ServingTenantQuotaError& e) {
                    // Race-path 429 (pre-check in handle() sheds the rest).
                    this->_metrics.recordQuotaRejected(state->tenantId);
                    errorMessage = e.what();
                } catch (const runtime::serving::ServingOverloadedError& e) {
                    this->_metrics.recordOverloadRejected(state->tenantId);
                    errorMessage = e.what();
                } catch (const std::exception& e) {
                    this->_metrics.recordError(state->tenantId);
                    errorMessage = e.what();
                }
            } else {
                std::lock_guard<std::mutex> lk{*targetMutex};
                try {
                    if (targetSpec != nullptr) {
                        generated = targetSpec->generate(
                            state->promptIds, state->params,
                            onToken, &stats,
                            onPrefillDone, onPrefillProgress);
                    } else {
                        generated = targetEng.generate(state->promptIds,
                                                    state->params,
                                                    onToken,
                                                    &stats,
                                                    onPrefillDone,
                                                    onPrefillProgress);
                    }
                } catch (const std::exception& e) {
                    this->_metrics.recordError(state->tenantId);
                    errorMessage = e.what();
                }
            }

            if (clientGone) {
                // Distinguish prefill vs decode abort in the log so the
                // operator can spot Pegenaut cancels happening during
                // the (potentially long) prefill phase — those are the
                // highest-value ones to see, they show the user gave up
                // while waiting for the first token.
                const bool cancelledInPrefill = (emittedTokens == 0);
                MM_LOG_INFO("server",
                            "stream {}: aborted mid-{} "
                            "(emitted={}, prefill={:.0f}ms)",
                            state->respId,
                            cancelledInPrefill ? "prefill" : "decode",
                            emittedTokens, stats.prefillMs);
                sink.done();
                return false;
            }

            if (!errorMessage.empty()) {
                // Surface the error as an SSE event so the client sees
                // a structured failure rather than just a half-stream.
                json errChunk = SseEncoder::streamChunkSkeleton(state->respId,
                                                    state->created,
                                                    state->echoModel);
                errChunk["error"] = json{
                    {"message", errorMessage},
                    {"type",    "server_error"},
                };
                (void)SseEncoder::writeSseEvent(sink, errChunk);
                MM_LOG_ERROR("server",
                             "stream {}: generate failed: {}",
                             state->respId, errorMessage);
                sink.done();
                return false;
            }

            // M-FunctionCalling: a call still open here means generation
            // ended without a closer reaching either the detector (Qwen) or
            // isStop (Gemma 4) — most likely max_tokens hit mid-call. Finalize
            // best-effort, mirroring ToolCallParser's tolerance for an
            // unterminated final block, rather than silently dropping it.
            // Any text held back purely as a defensive tail against a marker
            // split across tokens (and that never became part of a call) is
            // ordinary trailing content — fold it into the normal UTF-8 flush
            // below instead of a separate write.
            if (state->toolCallsEnabled) {
                if (state->toolCallDetector.buffering()) {
                    state->toolCallDetector.closeOnStop();
                    (void)emitToolCallBlock(state->toolCallDetector.completedBlock());
                    state->toolCallDetector.reset();
                }
                state->utf8Pending.append(state->toolCallDetector.takePendingFlush());
            }

            // Flush any leftover reasoning buffer first (a model that stops
            // while still "thinking" — e.g. length cutoff mid-<think> — should
            // still surface what it reasoned).
            if (!state->reasoningPending.empty()) {
                (void)SseEncoder::writeSseEvent(
                    sink,
                    SseEncoder::buildReasoningChunk(state->respId, state->created,
                                      state->echoModel, state->reasoningPending));
                state->reasoningPending.clear();
            }

            // Flush any UTF-8 buffer leftover (partial codepoint at the
            // end — rare but possible when the model stops mid-grapheme
            // on length cutoff). Receivers may show U+FFFD.
            if (!state->utf8Pending.empty()) {
                (void)SseEncoder::writeSseEvent(
                    sink,
                    SseEncoder::buildContentChunk(state->respId, state->created,
                                      state->echoModel,
                                      state->utf8Pending));
                state->utf8Pending.clear();
            }

            // 3. Finish chunk + DONE sentinel. finish_reason is "stop"
            //    if a stop token broke the decode loop, "length" if we
            //    exhausted maxNewTokens.
            const bool hitStop = stats.hitStop
                || (!generated.empty() && isStop(generated.back()));
            // M-FunctionCalling: matches the blocking path — a turn that
            // emitted a tool call reports "tool_calls" regardless of which
            // stop condition ended decoding.
            const std::string finish = state->anyToolCallEmitted ? "tool_calls"
                                      : (hitStop ? "stop" : "length");

            (void)SseEncoder::writeSseEvent(
                sink,
                SseEncoder::buildFinishChunk(state->respId, state->created,
                                 state->echoModel, finish));
            (void)SseEncoder::writeSseDone(sink);

            std::string streamSpecSuffix;
            if (stats.specDecRounds > 0 && stats.specDecDrafted > 0) {
                const double acceptRate =
                    static_cast<double>(stats.specDecAccepted)
                  / static_cast<double>(stats.specDecDrafted);
                streamSpecSuffix =
                      " spec_rounds=" + std::to_string(stats.specDecRounds)
                    + " spec_acc="   + std::to_string(stats.specDecAccepted)
                    + "/"            + std::to_string(stats.specDecDrafted)
                    + " spec_rate="  + std::to_string(acceptRate);
            }
            MM_LOG_INFO("server",
                        "stream {} model={} prompt={} cached={} emitted={} "
                        "prefill={:.0f}ms decode={:.0f}ms energy={:.1f}J "
                        "finish={}{}",
                        state->respId, state->echoModel,
                        state->promptIds.size(), stats.cachedTokens,
                        emittedTokens,
                        stats.prefillMs, stats.decodeMs,
                        stats.packageJoules, finish, streamSpecSuffix);

            this->_metrics.recordSuccess(
                state->tenantId, state->promptIds.size(), emittedTokens,
                stats.packageJoules, stats.prefillMs, stats.decodeMs);

            sink.done();
            return false;
        });
}

} // namespace mimirmind::server