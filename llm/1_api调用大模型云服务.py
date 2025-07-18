# -*- coding: utf-8 -*-
"""
1. 魔搭社区提供了符合 OpenAI大模型接口规范 的示例。
2.
"""

# 调用符合OpenAI规范的LLM云接口,流式返回
def call_api_by_openai():
    from openai import OpenAI

    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='64a90e7d-5e59-4d07-b2c7-382c29ff5256',  # ModelScope Token
    )

    # set extra_body for thinking control
    extra_body = {
        # enable thinking, set to False to disable
        "enable_thinking": True,
        # use thinking_budget to contorl num of tokens used for thinking
        # "thinking_budget": 4096
    }

    response = client.chat.completions.create(
        model='Qwen/Qwen3-0.6B',  # ModelScope Model-Id
        messages=[
            {
                'role': 'user',
                'content': '9.9和9.11谁大'
            }
        ],
        stream=True,
        extra_body=extra_body
    )
    done_thinking = False
    for chunk in response:
        thinking_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if thinking_chunk != '':
            print(thinking_chunk, end='', flush=True)
        elif answer_chunk != '':
            if not done_thinking:
                print('\n\n === Final Answer ===\n')
                done_thinking = True
            print(answer_chunk, end='', flush=True)


# 调用本地模型
def call_api_by_transformer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "model_local_save_path"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" # 自动加载模型，有gpu就gpu，没有就cpu，gpu一块不够就将模型切分多块放在不同gpu
    )

    # prepare the model input
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # the result will begin with thinking content in <think></think> tags, followed by the actual response
    print(tokenizer.decode(output_ids, skip_special_tokens=True))



if __name__ == '__main__':
    call_api_by_openai()

