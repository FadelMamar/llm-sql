from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from transformers import AutoTokenizer


class ModelManager:
    """Class to handle model-related operations."""

    @staticmethod
    def get_ollama_models() -> List[str]:
        """Get list of available Ollama models."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return [line.split()[0] for line in result.stdout.strip().split("\n")[1:]]
        except Exception:
            return []

    @staticmethod
    def load_ollama_llm(llm_name:str,
                        context_window:int=1024,
                        temperature:float=0.2, 
                        request_timeout:int=int(2e4)) -> Ollama:
        return Ollama(model=llm_name, 
                      request_timeout=request_timeout,
                      temperature=temperature,
                      context_window=context_window
                      )
    
    @staticmethod
    def load_gemini_llm(llm_name:str='models/gemini-1.5-flash',
                        num_output:int=128,system_prompt=None,
                        temperature:float=0.2):
        return Gemini(model=llm_name,
                      api_key=os.environ["GEMINI_API_KEY"],
                      system_prompt=system_prompt,
                      max_tokens=num_output,
                      temperature=temperature,
                      )
    
    @staticmethod
    def load_embed_model(model_id_or_path: str,
                         max_length:int, 
                         is_openvino:bool=False,
                         device='cpu',
                         cache_dir:str=None,
                         embed_batch_size:int=32) -> BaseEmbedding:
        
        """Load embedding model."""
        if is_openvino:
            return OpenVINOEmbedding(model_id_or_path=model_id_or_path,
                                    query_instruction=None, #TODO: understand
                                    text_instruction=None, #TODO: understand
                                    device='gpu',
                                    max_length=max_length,
                                    embed_batch_size=embed_batch_size,
                                    )
        
        return HuggingFaceEmbedding(model_name=model_id_or_path,
                                    max_length=max_length,
                                    query_instruction=None, #TODO: understand
                                    text_instruction=None, #TODO: understand
                                    embed_batch_size=embed_batch_size,
                                    cache_folder=cache_dir,
                                    device=device
                                )

    @staticmethod
    def load_tokenizer(huggingface_name:str|None,max_length:int):

        if huggingface_name is not None:
            return AutoTokenizer.from_pretrained(huggingface_name,
                                                 model_max_length=max_length)
        else:
            raise NotImplementedError
    


