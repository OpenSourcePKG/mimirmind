// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ChatCompletionHandler.hpp"

#include "server/ApiHelpers.hpp"
#include "server/ChatRequestParser.hpp"
#include "server/PromptTrimmer.hpp"
#include "server/RequestDispatcher.hpp"
#include "server/RequestTracker.hpp"
#include "server/SseEncoder.hpp"

#include "model/ResponseCleaner.hpp"
#include "model/Tokenizer.hpp"
#include "core/log/Log.hpp"
#include "runtime/SpeculativeDecoder.hpp"
#include "runtime/ThermalGuard.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>

namespace mimirmind::server {

using nlohmann::json;

ChatCompletionHandler::ChatCompletionHandler(RequestDispatcher&        dispatcher,
                                              RequestTracker&           tracker,
                                              model::ChatTemplate::Style chatStyle,
                                              const ServerConfig&        cfg)
    : _dispatcher{dispatcher},
      _tracker{tracker},
      _chatStyle{chatStyle},
      _cfg{cfg} {}

bool ChatCompletionHandler::prepareChatRequest(
    runtime::InferenceEngine&      targetEngine,
    const ChatRequest&             cr,
    httplib::Response&             res,
    std::vector<std::int32_t>&     promptIds,
    std::vector<std::int32_t>&     stopIds,
    runtime::GenerateParams&       params,
    TrimReport&                    report) {
    if (cr.messages.empty()) {
        sendError(res, 400, "invalid_request_error",
                  "messages must not be empty");
        return false;
    }

    const auto& tok = targetEngine.tokenizer();
    // M-PT: work on a mutable copy so the trim loop can drop entries
    // without touching the parsed request. Also lets us report the
    // original prompt-token count in the response.
    std::vector<model::ChatMessage> msgs = cr.messages;
    promptIds = model::ChatTemplate::encode(
        _chatStyle, tok, msgs, /*addGenerationPrompt=*/true);

    stopIds = model::ChatTemplate::stopIds(_chatStyle, tok);
    PromptTrimmer::extendStopIds(tok, cr.stopStrings, stopIds);

    params.maxNewTokens = cr.maxTokens > 0 ? cr.maxTokens : _cfg.defaultMaxNew;
    params.stopIds      = stopIds;

    // M-PT — server-side length discipline.
    {
        std::string trimErr;
        if (!PromptTrimmer::applyPromptTrim(msgs, promptIds, params.maxNewTokens,
                             targetEngine.maxContextTokens(),
                             targetEngine.config().contextLength,
                             tok, _chatStyle, report, trimErr)) {
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
    if (cr.hasTemperature) {
        params.sampling.temperature = cr.temperature;
    }
    params.sampling.topP = cr.topP;
    params.sampling.topK = cr.topK;
    params.sampling.seed = cr.seed;

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
    // per-engine thermal separation isn't a thing (single iGPU).
    if (auto* guard = _dispatcher.defaultEngine().thermalGuard(); guard != nullptr) {
        try {
            guard->checkAdmission();
        } catch (const runtime::ThermalLimitExceeded& e) {
            MM_LOG_INFO("server",
                        "thermal refusal: {}", e.what());
            res.set_header("Retry-After", "10");
            sendError(res, 503, "service_unavailable", e.what());
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
    if (!prepareChatRequest(engine, cr, res, promptIds, stopIds, params,
                            trimReport)) {
        return;
    }

    const auto& tok = engine.tokenizer();

    runtime::GenerateStats stats;
    std::vector<std::int32_t> generated;

    // Reserve the response id up-front so the /v1/system/status
    // snapshot can carry it while the request is still running.
    const std::string respId = makeRequestId();
    _tracker.begin(respId, promptIds.size(),
                   params.maxNewTokens, /*streaming=*/false);
    RequestTracker::Guard requestGuard{&_tracker};

    auto onPrefillProgress =
        [this](const runtime::InferenceEngine::PrefillProgress& p)
            -> bool {
            _tracker.updatePrefillProgress(p.blocksDone, p.blocksTotal, p.elapsedMs);
            return true;
        };
    auto onPrefillDone =
        [this](const runtime::InferenceEngine::PrefillDone&) {
            _tracker.markPrefillDone();
        };
    auto onToken = [this](std::int32_t) -> bool {
        _tracker.incrementDecodeTokens();
        return true;
    };

    {
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
            res.set_header("Retry-After", "10");
            sendError(res, 503, "service_unavailable", e.what());
            return;
        } catch (const std::exception& e) {
            MM_LOG_ERROR("server", "generate failed: {}", e.what());
            sendError(res, 500, "server_error",
                      std::string{"generate: "} + e.what());
            return;
        }
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
    const std::string text     = _cfg.preserveThinking
        ? rawText
        : model::ChatTemplate::cleanResponse(_chatStyle, rawText);

    const std::int64_t now   = unixNow();
    const std::string finish = hitStop ? "stop" : "length";

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

    json response = {
        {"id",      respId},
        {"object",  "chat.completion"},
        {"created", now},
        {"model",   echoModel},
        {"choices", json::array({
            json{
                {"index", 0},
                {"message", {
                    {"role",    "assistant"},
                    {"content", text},
                }},
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
    if (!prepareChatRequest(engine, cr, res, promptIds, stopIds, params,
                            trimReport)) {
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
        // Buffers a trailing incomplete UTF-8 codepoint between tokens
        // so SSE deltas always carry valid UTF-8.
        std::string                   utf8Pending;
        // Per-stream filter that swallows the Gemma 4
        // <|channel>thought<channel|> wrapper at the token level,
        // matching the behaviour ChatTemplate::cleanResponse applies in
        // the non-streaming path. No-op for other chat styles.
        model::ResponseCleaner        cleaner;
        bool                          done{false};

        StreamState(model::ChatTemplate::Style style,
                    const model::Tokenizer&    tok,
                    bool                       preserveThinking)
            : cleaner{preserveThinking
                ? model::ResponseCleaner{
                      model::ChatTemplate::Style::QwenChatML, -1, -1}
                : model::ResponseCleaner::forStyle(style, tok)} {}
    };
    auto state = std::make_shared<StreamState>(_chatStyle, engine.tokenizer(),
                                               _cfg.preserveThinking);
    state->promptIds = std::move(promptIds);
    state->stopIds   = std::move(stopIds);
    state->params    = std::move(params);
    state->respId    = respId;
    state->created   = created;
    state->echoModel = echoModel;

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
                    return true;
                }
                // Snapshot the per-token progress even for stripped
                // tokens so /v1/system/status reflects real decode work.
                _tracker.incrementDecodeTokens();
                std::string txt = tok.decode(
                    std::span<const std::int32_t>{&id, 1},
                    /*skipSpecial=*/true);
                if (txt.empty()) {
                    return true;
                }

                // Drop structural markup (Gemma 4 channel wrapper).
                if (!state->cleaner.feed(id, txt)) {
                    return true;
                }

                state->utf8Pending.append(txt);
                const std::size_t cut =
                    SseEncoder::utf8IncompleteTailStart(state->utf8Pending);
                if (cut == 0) {
                    return true;     // entire buffer is partial; keep waiting
                }

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
                return true;
            };

            // Named SSE event fired between prefill and the first
            // decode token so a streaming client can flip its UX from
            // "reading your prompt" to "answering". The OpenAI stream
            // demuxer ignores named events, browsers pick it up via
            // EventSource.addEventListener.
            auto onPrefillDone =
                [&](const runtime::InferenceEngine::PrefillDone& p) {
                    _tracker.markPrefillDone();
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
                    _tracker.updatePrefillProgress(p.blocksDone, p.blocksTotal, p.elapsedMs);
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
            RequestTracker::Guard requestGuard{&_tracker};
            {
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
            const std::string finish = hitStop ? "stop" : "length";

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

            sink.done();
            return false;
        });
}

} // namespace mimirmind::server