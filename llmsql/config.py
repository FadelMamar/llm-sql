from pathlib import Path
from dataclasses import dataclass
from llama_index.core import Settings
from .llm import ModelManager
from llama_index.core.callbacks import CallbackManager
from typing import Sequence


@dataclass
class Arguments:

    # inference
    device:str='cpu' # cuda, torch.device
    is_openvino_embed:bool=False

    # embeddings
    embed_cache_dir:str=r"D:\llm-sql\models"
    embed_batch_size:int=32

    # system prompt
    sys_prompt:str=None

    # db
    db_name:str="dvdrental"
    user_name:str="postgres"
    pwd:str=""
    host:str="localhost"
    port:str="5432"

    # llm
    llm_type:str='gemini' # "gemini", "ollama", "deepseek-r1:7b"
    tokenizer_name:str='Qwen/Qwen2.5-7B-Instruct' # deepseek-ai/DeepSeek-R1 HuggingFaceTB/SmolLM-1.7B-Instruct
    llm_name:str='models/gemini-1.5-flash' # "models/gemini-1.5-flash" "qwen2.5:7b" "smollm:latest" "Qwen/Qwen2.5-7B-Instruct"
    embed_model:str=r"D:\llm-sql\models\kalm_ov" # "D:\llm-sql\models\HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1" or "D:\llm-sql\models\kalm_ov"
    context_window:int=8000
    num_output:int=256
    temperature:float=0.3
    timeout:int=int(2e4)



class SettingsManager:
    
    @classmethod
    def set_settings(cls,
                     args:Arguments,
                     callbackhandlers:list=[]):
        
        # get llm
        if args.llm_type == 'ollama':
            Settings.llm = ModelManager.load_ollama_llm(args.llm_name,
                                                        context_window=args.context_window,
                                                        temperature=args.temperature,
                                                        request_timeout=args.timeout)
        elif args.llm_type == 'gemini':
            Settings.llm = ModelManager.load_gemini_llm(llm_name=args.llm_name,
                                                        system_prompt=args.sys_prompt,
                                                        num_output=args.num_output,
                                                        temperature=args.temperature)
        else:
            raise NotImplementedError
        Settings.tokenizer = ModelManager.load_tokenizer(huggingface_name=args.tokenizer_name,
                                                         max_length=args.context_window)

        # llm
        Settings.embed_model = ModelManager.load_embed_model(model_id_or_path=args.embed_model,
                                                             max_length=args.context_window,
                                                             embed_batch_size=args.embed_batch_size,
                                                             is_openvino=args.is_openvino_embed,
                                                             cache_dir=args.embed_cache_dir,
                                                             device=args.device
                                                            )

        # i/o to llm
        Settings.num_output = args.num_output
        Settings.context_window = args.context_window

        # prompts
        # Settings._prompt_helper
        Settings.callback_manager = CallbackManager(handlers=callbackhandlers)