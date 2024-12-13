#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import os
import random
import unittest
from pprint import pprint

import coremltools as ct
import torch
from argmaxtools import _sdpa, compress
from argmaxtools import test_utils as argmaxtools_test_utils
from argmaxtools.utils import get_fastest_device, get_logger
from tqdm import tqdm
from transformers import AutoTokenizer, WhisperForConditionalGeneration
from transformers.models.whisper import modeling_whisper

from whisperkit import test_utils, text_decoder
from whisperkit.compress import palettize

torch.set_grad_enabled(False)
logger = get_logger(__name__)

# Test constants
TEST_WHISPER_VERSION = (
    os.getenv("TEST_WHISPER_VERSION", None) or "openai/whisper-tiny"
)  # tiny"
TEST_DEV = os.getenv("TEST_DEV", None) or get_fastest_device()
TEST_TORCH_DTYPE = torch.float32
TEST_PSNR_THR = 35
TEST_CACHE_DIR = os.getenv("TEST_CACHE_DIR", None) or "/tmp"

argmaxtools_test_utils.TEST_SKIP_SPEED_TESTS = True

# WhisperDecoderContextPrefill constants
TEST_PREFILL_CONSISTENCY_PSNR_THR = 20
TEST_BATCH = 16
TEST_OUTPUT_NAMES = [
    "logits", "key_cache_updates", "value_cache_updates", "alignment_heads_weights"]
TEST_CONTEXT_PREFILL_OUTPUT_NAMES = ["key_cache_prefill", "value_cache_prefill"]
TEST_DEC_KV_SEQ_LEN = None
TEST_TOKEN_TIMESTAMPS = True


def load_whisper_model(model_path: str, torch_dtype=None):
    """Load a Whisper model from either Hugging Face hub or local path
    
    Args:
        model_path: Either a Hugging Face model ID or local directory path
        torch_dtype: Optional torch dtype to load the model in
    
    Returns:
        The loaded Whisper model
    """
    try:
        # First try loading as a local path
        if os.path.exists(model_path):
            return WhisperForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch_dtype
            )
        # If not a valid path, try loading from HF hub
        return WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype
        )
    except Exception as e:
        raise ValueError(
            f"Could not load model from '{model_path}'. "
            "Make sure it is either a valid local path or Hugging Face model ID."
        ) from e


class TestWhisperTextDecoder(argmaxtools_test_utils.CoreMLTestsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_output_names = TEST_OUTPUT_NAMES
        cls.test_cache_dir = TEST_CACHE_DIR
        cls.model_name = "TextDecoder"

        if not TEST_TOKEN_TIMESTAMPS:
            cls.test_output_names.pop(cls.test_output_names.index("alignment_heads_weights"))

        # Original model
        orig_torch_model = load_whisper_model(TEST_WHISPER_VERSION, TEST_TORCH_DTYPE)
        cls.orig_torch_model = (
            orig_torch_model.model.decoder.to(TEST_DEV).to(TEST_TORCH_DTYPE).eval()
        )

        # Base test model
        cls.test_torch_model = text_decoder.WhisperTextDecoder(
            cls.orig_torch_model.config
        )
        cls.test_torch_model.load_state_dict(cls.orig_torch_model.state_dict())
        cls.test_torch_model = (
            cls.test_torch_model.to(TEST_DEV).to(TEST_TORCH_DTYPE).eval()
        )
        cls.gen_cfg = orig_torch_model.generation_config

        if TEST_TOKEN_TIMESTAMPS:
            cls.test_torch_model.configure_for_token_timestamps(cls.gen_cfg)

        # Elaboration: I/O and architecture config
        cfg = cls.orig_torch_model.config
        cls.cfg = dict(
            n_heads=cfg.decoder_attention_heads,
            n_layers=cfg.decoder_layers,
            embed_dim=cfg.d_model,
            batch_size=1,
            vocab_size=cfg.vocab_size,
            enc_seq_len=cfg.max_source_positions,
            dec_kv_seq_len=TEST_DEC_KV_SEQ_LEN or cfg.max_target_positions,
        )

        cls.cfg["active_dec_kv_seq_len"] = random.randint(1, cls.cfg["dec_kv_seq_len"])
        logger.info(
            f"Decoding token at index {cls.cfg['active_dec_kv_seq_len']} "
            f"(max={cls.cfg['dec_kv_seq_len']}) "
        )

        (
            cls.test_torch_inputs,
            cls.orig_inputs,
        ) = test_utils._prepare_test_inputs_for_decoder(**cls.cfg)

        # Do casting and device placement per test config
        cls.test_torch_inputs = {k: place(v) for k, v in cls.test_torch_inputs.items()}
        cls.orig_inputs = {
            k: place(v)
            if isinstance(v, torch.Tensor)
            else [(place(vi[0]), place(vi[1])) for vi in v]
            for k, v in cls.orig_inputs.items()
        }

        # Cache torch outputs for later comparison
        W = cls.orig_torch_model.embed_tokens.weight.T
        cls.orig_torch_out_logits = (
            cls.orig_torch_model(**cls.orig_inputs)[0].squeeze() @ W
        )
        cls.orig_torch_output = cls.orig_torch_out_logits
        cls.orig_torch_out_logits_argmax = cls.orig_torch_out_logits.argmax()

        test_torch_out = cls.test_torch_model(**cls.test_torch_inputs)
        cls.test_torch_out_logits = test_torch_out[0].squeeze()
        cls.test_torch_output = cls.test_torch_out_logits
        cls.test_torch_out_logits_argmax = cls.test_torch_out_logits.argmax()

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # Models
        cls.orig_torch_model = None
        cls.test_torch_model = None
        cls.test_coreml_model = None

        # Data
        cls.cfg
        cls.gen_cfg = None
        cls.active_dec_kv_seq_len = None
        cls.orig_inputs = None
        cls.test_torch_inputs = None
        cls.orig_torch_out_logits = None
        cls.orig_torch_out_logits_argmax = None
        cls.test_torch_out_logits = None
        cls.test_torch_out_logits_argmax = None

        super().tearDownClass()

    def test_torch2torch_correctness(self):
        """Coverage:
        - torch2torch parity transformers.models.whisper.modeling_whisper.WhisperDecoder
        and whisperkit.text_decoder.WhisperTextDecoder
        """
        with self.subTest(phase="torch_correctness_probs"):
            psnr = argmaxtools_test_utils.compute_psnr(
                self.orig_torch_out_logits, self.test_torch_out_logits
            )

            logger.info(f"torch2torch probs PSNR={psnr:.3g}")
            self.assertGreater(psnr, TEST_PSNR_THR)

        with self.subTest(phase="torch_correctness_logits_argmax"):
            argmax_accuracy = (
                self.orig_torch_out_logits_argmax.eq(self.test_torch_out_logits_argmax)
                .float()
                .mean()
                .item()
                * 100.0
            )

            logger.info(f"torch2torch logits argmax accuracy={argmax_accuracy:.3g}%")
            self.assertEqual(argmax_accuracy, 100.0)

    def test_torch_context_prefill(self):
        """
        Coverage:
        - Build a WhisperDecoderContextPrefill module
        - Use a test WhisperDecoder instance to compute all valid context prefills
        - Test forward pass on WhisperDecoderContextPrefill
        """
        # Prepare additional inputs and utilities required for this test
        str2id = AutoTokenizer.from_pretrained(TEST_WHISPER_VERSION).vocab
        encoder_output_embeds_1 = torch.randn_like(
            self.test_torch_inputs["encoder_output_embeds"].expand(
                TEST_BATCH, -1, -1, -1
            )
        )
        encoder_output_embeds_2 = torch.randn_like(encoder_output_embeds_1)

        # Init two distinct estimations of the context cache prefills for consistency tests
        self.test_context_prefill = text_decoder.WhisperTextDecoderContextPrefill(
            self.test_torch_model, encoder_output_embeds_1
        )

        self.test_context_prefill_2 = text_decoder.WhisperTextDecoderContextPrefill(
            self.test_torch_model, encoder_output_embeds_2
        )

        # Compare outputs across implementations (Hugging Face and Argmax) as well as across
        # different batches of encoder outputs to validate the hypothesis that context prefill
        # caches are invariant to encoder-related decoder inputs (`encoder_output_embeds`)
        for task_idx, task_spec in tqdm(
            enumerate(self.test_context_prefill.valid_task_specs)
        ):
            # Retrieve test model's caches for current task_idx across two encoder input batches
            task, language = self.test_context_prefill.task_idx_to_task_and_language(
                torch.tensor(task_idx, dtype=torch.int32)
            )
            test_key_cache_1, test_value_cache_1 = [
                t.flatten() for t in self.test_context_prefill(task, language)
            ]
            test_key_cache_2, test_value_cache_2 = [
                t.flatten() for t in self.test_context_prefill_2(task, language)
            ]

            # Compute equivalent caches from the original model across two encoder input batches
            task_spec_ids = [
                str2id[token] for token in ("<|startoftranscript|>",) + task_spec
            ]
            ref_key_cache_1, ref_value_cache_1 = test_utils._get_context_prefill_from(
                hf_whisper_decoder=self.orig_torch_model,
                prefill_decoder_ids=task_spec_ids,
                encoder_output_embeds=encoder_output_embeds_1,
            )

            ref_key_cache_2, ref_value_cache_2 = test_utils._get_context_prefill_from(
                hf_whisper_decoder=self.orig_torch_model,
                prefill_decoder_ids=task_spec_ids,
                encoder_output_embeds=encoder_output_embeds_2,
            )

            psnr_fn = argmaxtools_test_utils.compute_psnr
            ref_vs_test_consistency = {
                "ref_vs_test_key_psnr_inp1_vs_inp1": psnr_fn(
                    ref_key_cache_1, test_key_cache_1
                ),
                "ref_vs_test_key_psnr_inp1_vs_inp2": psnr_fn(
                    ref_key_cache_1, test_key_cache_2
                ),
                "ref_vs_test_value_psnr_inp1_vs_inp1": psnr_fn(
                    ref_value_cache_1, test_value_cache_1
                ),
                "ref_vs_test_value_psnr_inp1_vs_inp2": psnr_fn(
                    ref_value_cache_1, test_value_cache_2
                ),
            }

            logger.info(
                "Decoder context prefill consistency across different implementations "
                "(ref vs test)"
            )

            pprint({k: f"{v:.3g}" for k, v in ref_vs_test_consistency.items()})
            for psnr in ref_vs_test_consistency.values():
                self.assertGreater(psnr, TEST_PREFILL_CONSISTENCY_PSNR_THR)

            test_vs_test_different_inputs_consistency = {
                "test_vs_test_key_psnr_inp1_vs_inp2": psnr_fn(
                    test_key_cache_1, test_key_cache_2
                ),
                "test_vs_test_value_psnr_inp1_vs_inp2": psnr_fn(
                    test_value_cache_1, test_value_cache_2
                ),
            }
            logger.info(
                "Decoder context prefill consistency across different inputs (same implementation)"
            )

            pprint(
                {
                    k: f"{v:.3g}"
                    for k, v in test_vs_test_different_inputs_consistency.items()
                }
            )
            for psnr in test_vs_test_different_inputs_consistency.values():
                self.assertGreater(psnr, TEST_PREFILL_CONSISTENCY_PSNR_THR)

        # Test input: Transcribe (task=0) English
        self.test_torch_inputs = dict(
            task=torch.tensor([0], dtype=torch.int32),
            language=torch.tensor(
                [self.test_context_prefill.tokenizer.vocab["<|en|>"]], dtype=torch.int32
            ),
        )

        self.test_coreml_inputs = argmaxtools_test_utils._create_coreml_inputs(
            self.test_torch_inputs
        )

        with self.subTest(phase="trace_torch"):
            self.traced_test_torch_model = torch.jit.trace(
                self.test_context_prefill,
                tuple(list(self.test_torch_inputs.values())),
            )

        with self.subTest(phase="coreml_conversion_and_correctness"):
            self.test_coreml_model = ct.convert(
                self.traced_test_torch_model,
                inputs=[
                    ct.TensorType(k, shape=v.shape, dtype=v.cpu().numpy().dtype)
                    for k, v in self.test_torch_inputs.items()
                ],
                outputs=[
                    ct.TensorType(
                        output_name,
                        dtype=argmaxtools_test_utils.TEST_COREML_IO_FLOAT_DTYPE,
                    )
                    for output_name in TEST_CONTEXT_PREFILL_OUTPUT_NAMES
                ],
                minimum_deployment_target=ct.target.macOS14,
                compute_units=argmaxtools_test_utils.TEST_COMPUTE_UNIT,
            )

            self.test_coreml_out = self.test_coreml_model.predict(
                self.test_coreml_inputs
            )
            self.test_torch_out = self.test_context_prefill(**self.test_torch_inputs)

            for oidx, output_name in enumerate(TEST_CONTEXT_PREFILL_OUTPUT_NAMES):
                psnr = psnr_fn(
                    self.test_torch_out[oidx],
                    torch.from_numpy(self.test_coreml_out[output_name]).to(TEST_DEV),
                )
                logger.info(f"torch2coreml {output_name}: PSNR={psnr:.3g}")
                self.assertGreater(psnr, TEST_PSNR_THR)

            argmaxtools_test_utils._save_coreml_asset(
                self.test_coreml_model, TEST_CACHE_DIR, "TextDecoderContextPrefill"
            )


argmaxtools_test_utils.TEST_DONT_PALETTIZE_TOP_K = 0
argmaxtools_test_utils.TEST_ALLOWED_NBITS = [4, 6, 8]
compress.palettize.NUM_MIXED_BIT_RECIPES = 1
compress.palettize.TEST_BATCH_SIZE = 16
compress.palettize.INVERTED_RESULT_THR = 0.25
compress.palettize.SPARSE_OUTLIER_DECOMPOSITION = False
compress.sparse_outlier.OUTLIER_NUM_STD = 3.0


class TestWhisperTextDecoderPalettizer(
    argmaxtools_test_utils.CoreMLPalettizerTestsMixin, unittest.TestCase
):
    """
    Unit tests for whisperkit.compress.palettize.kitWhisperDecoderPalettizer

    Coverage:
    - Per-layer palettization
    - Cumulative palettization
    - Mixed-bit palettization
    - Core ML model compression and correctness

    """

    @classmethod
    def setUpClass(cls):
        cls.model_name = "TextDecoder"
        cls.output_names = TEST_OUTPUT_NAMES
        if not TEST_TOKEN_TIMESTAMPS:
            cls.output_names.pop("alignment_heads_weights")

        cls.palettizer = palettize.WhisperTextDecoderPalettizer(
            model_version=TEST_WHISPER_VERSION,
            cache_dir=os.path.join(
                TEST_CACHE_DIR, "compression_artifacts", "TextDecoder"
            ),
        )

    @classmethod
    def tearDownClass(cls):
        cls.output_names = None
        cls.palettizer = None


# Helpers
def place(t):
    return (
        t.to(TEST_DEV).to(TEST_TORCH_DTYPE)
        if t.dtype.is_floating_point
        else t.to(TEST_DEV)
    )


def main(args):
    global TEST_WHISPER_VERSION, TEST_CACHE_DIR, TEST_DEC_KV_SEQ_LEN, TEST_TOKEN_TIMESTAMPS

    TEST_WHISPER_VERSION = args.test_model_version
    TEST_TOKEN_TIMESTAMPS = not args.disable_token_timestamps

    logger.info(f"Testing {TEST_WHISPER_VERSION}")

    text_decoder.SDPA_IMPL = getattr(_sdpa, args.sdpa_implementation)
    logger.info(f"Set SDPA implementation to: {text_decoder.SDPA_IMPL}")

    if args.test_seq_len is not None:
        TEST_DEC_KV_SEQ_LEN = args.test_seq_len
        test_utils.TEST_DEC_KV_SEQ_LEN = args.test_seq_len
        logger.info(f"Overriding default sequence length to {args.test_seq_len}")

    with argmaxtools_test_utils._get_test_cache_dir(
        args.persistent_cache_dir
    ) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()

        if not args.disable_default_tests:
            suite.addTest(TestWhisperTextDecoder("test_torch2torch_correctness"))
            suite.addTest(
                TestWhisperTextDecoder("test_torch2coreml_correctness_and_speedup")
            )
        else:
            logger.info("Skipped default tests")

        if args.context_prefill_tests:
            suite.addTest(TestWhisperTextDecoder("test_torch_context_prefill"))

        if args.palettizer_tests:
            suite.addTest(TestWhisperTextDecoderPalettizer("test_profile_response"))
            suite.addTest(
                TestWhisperTextDecoderPalettizer(
                    "test_palettized_torch2coreml_conversion_and_correctness"
                )
            )

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-cache-dir", default=None, type=str)
    parser.add_argument("--palettizer-tests", action="store_true")
    parser.add_argument("--disable-default-tests", action="store_true")
    parser.add_argument("--context-prefill-tests", action="store_true")
    parser.add_argument("--disable-token-timestamps", action="store_true")
    parser.add_argument(
        "--sdpa-implementation", default="Cat", choices=tuple(_sdpa.__all__)
    )
    parser.add_argument(
        "--test-model-version",
        default=TEST_WHISPER_VERSION,
    )
    parser.add_argument(
        "--test-seq-len",
        default=None,
        type=int,
        help="Overrides model's default sequence length",
    )
    args = parser.parse_args()

    main(args)
