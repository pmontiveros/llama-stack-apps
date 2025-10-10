# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from llama_stack_client import LlamaStackClient
from termcolor import colored


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"
    return data_url

def check_model_is_available(client: LlamaStackClient, model: str):
    available_models = [
        model.identifier
        for model in client.models.list()
        if model.model_type == "llm" and "guard" not in model.identifier
    ]

    if model not in available_models:
        print(
            colored(
                f"Model `{model}` not found. Available models:\n\n{available_models}\n",
                "red",
            )
        )
        return False

    return True


def get_any_available_model(client: LlamaStackClient):
    available_models = [
        model.identifier
        for model in client.models.list()
        if model.model_type == "llm" and "guard" not in model.identifier
    ]
    if not available_models:
        print(colored("No available models.", "red"))
        return None

    return available_models[0]