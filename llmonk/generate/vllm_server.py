"""
Modified from https://github.com/vllm-project/vllm/blob/a132435204aac8506e41813f90d08ddf7eca43b2/vllm/entrypoints/api_server.py
"""

import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.inputs import TokensPrompt
import torch

from dataclasses import fields
from llmonk.utils import dataclass_to_dict

TIMEOUT_KEEP_ALIVE = 18000  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None
import random


def get_masked_experts(num_layers=12, experts_per_layer=64, percentage_to_mask=0.10, random_seed=42):
    # Calculate the number of experts to mask per layer
    # set random seed
    random.seed(random_seed)

    num_to_mask = round(experts_per_layer * percentage_to_mask)

    # Initialize the expert_masks dictionary
    expert_masks = {}

    # Populate the dictionary with random masks for each layer
    for layer in range(num_layers):
        # Randomly select experts to mask
        masked_experts = random.sample(range(experts_per_layer), num_to_mask)
        # Add the layer and its masked experts to the dictionary
        expert_masks[f'model.layers.{layer}.mlp'] = masked_experts

    # Print the resulting expert_masks dictionary
    for layer, masked_experts in expert_masks.items():
        print(f"{layer}: {masked_experts}")
    return expert_masks

def make_output(request_output: RequestOutput):
    out = {}
    # iterate over dataclass fields
    for field in fields(CompletionOutput):
        # get the field name
        field_name = field.name
        field_list = [getattr(o, field_name) for o in request_output.outputs]
        out[field_name] = field_list

    if out["logprobs"][0] is not None:
        condensed_logprobs = []
        for old_logprobs in out["logprobs"]:
            new_logprobs = []
            for logprob in old_logprobs:
                new_logprobs.append({k: v.logprob for k, v in logprob.items()})
            condensed_logprobs.append(new_logprobs)

        out["logprobs"] = condensed_logprobs

    out["request_id"] = request_output.request_id
    out["prompt"] = request_output.prompt
    out["prompt_token_ids"] = request_output.prompt_token_ids
    out["prompt_logprobs"] = request_output.prompt_logprobs

    return dataclass_to_dict(out)


@app.get("/ping")
async def ping() -> Response:
    """Ping the server."""
    return Response(status_code=200, content="pong")


@app.get("/max_batch_size")
async def max_batch_size() -> Response:
    """Get the maximum batch size."""
    return JSONResponse({"max_batch_size": engine_args.max_num_seqs})


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt", None)
    input_ids = request_dict.pop("input_ids", None)

    if prompt is None and input_ids is None:
        return Response(
            status_code=400, content="Prompt or input_ids must be provided."
        )

    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    if prompt is None:
        prompt = TokensPrompt(prompt_token_ids=input_ids)

    results_generator = engine.generate(
        prompt, sampling_params=sampling_params, request_id=request_id
    )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            out = make_output(request_output)
            yield (json.dumps(out) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)

        final_output = request_output

    assert final_output is not None
    out = make_output(final_output)
    return JSONResponse(out)

def apply_expert_masks(model, expert_masks):
    """
    Apply masks to experts in MoE layers by setting gate logits to a large negative value for masked experts.

    Args:
        model (torch.nn.Module): The model containing MoE layers.
        expert_masks (dict): A dictionary where keys are layer identifiers and values are lists of expert indices to mask.
    """
    LARGE_NEGATIVE = -10000

    for name, module in model.named_modules():
        if 'layers.' in name and '.mlp' in name:
            layer_num = name.split('layers.')[1].split('.')[0]
            layer_key = f'model.layers.{layer_num}.mlp'
            if layer_key in expert_masks:
                if hasattr(module, 'gate'):
                    indices_to_mask = expert_masks[layer_key]

                    def make_gate_forward_hook(indices):
                        def hook(module, input, output):
                            router_logits, router_weights = output
                            # Convert indices to tensor
                            mask_indices = torch.tensor(indices, device="cuda:0")
                            # Create a mask of the same shape as router_logits
                            mask = torch.zeros_like(router_logits)
                            # Set the masked indices to LARGE_NEGATIVE
                            mask[:, mask_indices] = LARGE_NEGATIVE
                            # Return modified tuple
                            return (router_logits + mask, router_weights)
                        return hook

                    module.gate.register_forward_hook(make_gate_forward_hook(indices_to_mask))
                    print(f"Registered mask for {layer_key} experts {indices_to_mask}")

def verify_expert_masks_applied(model, expert_masks):
    """Verify that forward hooks have been registered for the specified experts in MoE layers."""
    for name, module in model.named_modules():
        if 'layers.' in name and '.mlp' in name:
            layer_num = name.split('layers.')[1].split('.')[0]
            layer_key = f'model.layers.{layer_num}.mlp'
            if layer_key in expert_masks:
                if hasattr(module, 'gate'):
                    if hasattr(module.gate, '_forward_hooks'):
                        hooks = module.gate._forward_hooks
                        if hooks:
                            print(f"Forward hooks registered on {layer_key}.gate: {hooks}")
                        else:
                            print(f"No forward hooks registered on {layer_key}.gate.")
                    else:
                        print(f"{layer_key}.gate does not have '_forward_hooks' attribute.")
                else:
                    print(f"{layer_key} does not have a 'gate' attribute.")
            else:
                print(f"{layer_key} not in expert_masks.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Print engine structure
    print("\nEngine attributes:")
    print(dir(engine))

    print("\nEngine.engine.model_executor.driver_worker.model_runner attributes:")
    print(dir(engine.engine.model_executor.driver_worker.model_runner))

    print("\nModel attributes:")
    print(dir(engine.engine.model_executor.driver_worker.model_runner.model))

    # Print model structure
    print("\nModel named modules:")
    for name, module in engine.engine.model_executor.driver_worker.model_runner.model.named_modules():
        print(f"Module: {name}")
        print(f"Type: {type(module)}")

    # get expert masks
    expert_masks = get_masked_experts()

    # Apply masks to the correct model path
    print("Applying expert masks:")
    apply_expert_masks(engine.engine.model_executor.driver_worker.model_runner.model, expert_masks)
    verify_expert_masks_applied(engine.engine.model_executor.driver_worker.model_runner.model, expert_masks)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
