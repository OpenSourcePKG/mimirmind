// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

// Fwd-decls only — the impl .cpp pulls in the concrete headers.
namespace mimirmind::runtime { class InferenceEngine; }
namespace mimirmind::core::l0 { class L0Context; class UsmAllocator; }
namespace mimirmind::core::gguf { class WeightsMap; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::diagnostics {

/**
 * The M1-M5 + M7c smoke suite that runs under `mimirmind smoke`.
 * Each function prints its own report block to stdout and (when a
 * kernel launch fails) logs the exception via MM_LOG_ERROR before
 * returning — none of them throw. They are called sequentially by
 * `runSmoke` in cli/SmokeMode.cpp; ordering matters because later
 * stages assume the earlier stages loaded the model.
 *
 * Functions grouped here:
 *   - M1/M2 device + USM probe printers  (printM1M2, printM3Summary)
 *   - M2b allocator smoke                (runM2bAllocatorSmoke)
 *   - M5* single-kernel parity           (rmsNorm, Q4_K, Q6_K matvec)
 *   - M4* model-level smoke              (embed + lm_head, generate)
 *   - M7c chat-template render           (runM7cChatTemplate)
 *
 * All were inlined in main.cpp before Phase 2 of the multi-backend
 * refactor. Extracted here so main.cpp stays a thin dispatcher.
 */

void printM1M2(::mimirmind::runtime::InferenceEngine& engine);
void printM3Summary(const ::mimirmind::runtime::InferenceEngine& engine);

void runM2bAllocatorSmoke(::mimirmind::core::l0::UsmAllocator& allocator);

void runM5RmsNormParity(::mimirmind::core::l0::L0Context&    ctx,
                        ::mimirmind::core::l0::UsmAllocator& allocator);

void runM5bQ4KParity(::mimirmind::core::l0::L0Context&              ctx,
                     ::mimirmind::core::l0::UsmAllocator&           allocator,
                     const ::mimirmind::core::gguf::WeightsMap&     weights);

void runM5cQ6KParity(::mimirmind::core::l0::L0Context&              ctx,
                     ::mimirmind::core::l0::UsmAllocator&           allocator,
                     const ::mimirmind::core::gguf::WeightsMap&     weights);

void runM4aEmbedAndM4bLmHead(::mimirmind::runtime::InferenceEngine& engine);

void runM7cChatTemplate(const ::mimirmind::runtime::InferenceEngine& engine);

void runM4deGenerate(::mimirmind::runtime::InferenceEngine& engine,
                     const ::mimirmind::cli::CliArgs&       args);

} // namespace mimirmind::diagnostics