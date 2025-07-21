import os
from lite_infer import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/home/my_ubuntu/AI_deployment/lite_infer/model/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
    prompts = [
        "introduce Shanghai, China",
        "list all prime numbers within 100",
        "introduce self-attention",
        "introduce Beijing, China",
        "give me some tip about how to learn python",
        "introduce Meta company",
        "give me some travelling tip about Germany",
        "give me some tip about relax body",
        "A Complete Introduction to the History of the American Civil War",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "Can you introduce the History of the American Civil War. ",
        "who is the first president of the United States and what's his life story?",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
        "How to learn cuda programming, give me some code example.",
        "How to learn rust, give me some code examples.",
        "How to learn java, give me some code example.",
        "How to learn linux c, give me some code examples.",
        "A Complete Introduction to the History of the American Civil War",
        "Python is a good programming language, how tolearn it?",
        "Please introduce llama model architecture and give implement cuda code.",
        "Please introduce Qwen2.5 model structure and give cuda implement code."

    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print("\n")
    print(f"actual_cached_token: {llm.scheduler.actual_cached_token}\n")
    print(f"actual_cached_block: {llm.scheduler.actual_cached_block}\n")
    print(f"actual_num_tokens: {llm.scheduler.actual_num_tokens}\n")
    print(f"actual_num_block: {llm.scheduler.actual_num_block}\n")



if __name__ == "__main__":
    main()
