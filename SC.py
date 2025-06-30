import os
import re
from collections import Counter
import openai

# --- 1. 准备工作：设置API客户端 ---
# 代码将自动从环境变量 "OPENAI_API_KEY" 中获取密钥
try:
    """
     deepseek平台
    model:
        deepseek-chat
        deepseek-reasoner
    """
    # model = 'deepseek-chat'
    # client = openai.OpenAI(api_key="sk-XXX", base_url="https://api.deepseek.com")
    
    # 硅基流动平台
    client = openai.OpenAI(api_key="sk-XXX", base_url="https://api.siliconflow.cn")
    model = 'Qwen/Qwen3-8B'
    
except openai.OpenAIError as e:
    print(f"请确保您已设置 OPENAI_API_KEY 环境变量。错误: {e}")
    exit()

# --- 2. 定义思维链（Chain-of-Thought）提示 ---
# 这个提示来自论文附录，包含几个引导模型进行逐步推理的示例
COT_PROMPT = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: When I was 6, my sister was half my age. Now I am 70, how old is my sister?
A:
"""

# 我们将要解决的问题
# 这是一个经典陷阱题，贪心解码很容易出错
QUESTION = "When I was 6, my sister was half my age. Now I am 70, how old is my sister?"


def parse_answer(text: str) -> str | None:
    """从模型的输出中解析出最终的数字答案。"""
    # 使用正则表达式寻找 "The answer is X" 这种模式
    match = re.search(r"The answer is (\d+)", text)
    if match:
        return match.group(1)
    
    # 作为备用方案，尝试寻找文本末尾的数字
    match = re.search(r"(\d+)$", text.strip())
    if match:
        return match.group(1)
        
    return None


def call_llm(full_prompt: str, n: int, temperature: float, model: str):
    """调用LLM API并返回生成的文本列表。"""
    if n==1:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=256,
                n=n,  # 生成 n 个独立的推理路径
                temperature=temperature, # 控制多样性
                top_p=1.0,
                stop=["\nQ:"] # 在下一个问题开始时停止生成
            )
            content = [choice.message.content.strip() for choice in response.choices]
            return content
        except openai.APIError as e:
            print(f"调用API时发生错误: {e}")
            return []
    else:
        try:
            choices = []
            for i in range(n):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=256,
                    n=1,  # 生成 n 个独立的推理路径
                    temperature=temperature, # 控制多样性
                    top_p=1.0,
                    stop=["\nQ:"] # 在下一个问题开始时停止生成
                )
                content = [choice.message.content.strip() for choice in response.choices]
                choices.append(content[0])
                
            return choices
        except openai.APIError as e:
            print(f"调用API时发生错误: {e}")
            return []        


def greedy_decoding(prompt: str, question: str, model: str = "gpt-3.5-turbo"):
    """
    基线方法：标准的CoT，使用贪心解码（temperature=0）。
    只生成一个推理路径。
    """
    print("--- 正在执行基线方法：贪心解码 (Greedy Decode) ---")
    full_prompt = prompt.replace("A:\n", f"A: {question}\n")
    
    # temperature=0 确保了输出是确定性的（贪心）
    # n=1 因为我们只需要一个路径
    completions = call_llm(full_prompt, n=1, temperature=0.0, model=model)
    
    if not completions:
        return "未能从API获取结果", None
        
    reasoning_path = completions[0]
    final_answer = parse_answer(reasoning_path)
    
    print(f"\n[生成的推理路径 (单一)]:\n{reasoning_path}")
    
    return final_answer, reasoning_path


def self_consistency_decoding(prompt: str, question: str, num_paths: int = 10, model: str = "gpt-3.5-turbo"):
    """
    核心方法：自洽性解码。
    通过采样多个路径并进行多数投票来找到最一致的答案。
    """
    print(f"\n--- 正在执行核心方法：自洽性解码 (Self-Consistency) [采样 {num_paths} 条路径] ---")
    full_prompt = prompt.replace("A:\n", f"A: {question}\n")
    
    # temperature > 0 (例如 0.7) 鼓励模型生成多样化的输出
    completions = call_llm(full_prompt, n=num_paths, temperature=0.7, model=model)
    
    if not completions:
        return "未能从API获取结果", {}

    print(f"\n[生成的 {len(completions)} 条多样化推理路径]:")
    parsed_answers = []
    reasoning_paths = {}

    for i, path in enumerate(completions):
        answer = parse_answer(path)
        print(f"\n路径 {i+1}: \n{path}")
        print(f"  --> 解析出的答案: {answer}")
        if answer:
            parsed_answers.append(answer)
            # 记录每条路径，方便后续查看
            if answer not in reasoning_paths:
                reasoning_paths[answer] = []
            reasoning_paths[answer].append(path)

    if not parsed_answers:
        return "在所有路径中都未能解析出答案", reasoning_paths
    
    # 使用Counter进行多数投票
    answer_counts = Counter(parsed_answers)
    most_common_answer = answer_counts.most_common(1)[0][0]
    
    return most_common_answer, reasoning_paths, answer_counts


if __name__ == "__main__":
    print(f"问题: {QUESTION}\n")
    
    
    # 1. 运行基线方法
    greedy_answer, _ = greedy_decoding(COT_PROMPT, QUESTION, model=model)
    print(f"\n>>> 贪心解码的最终答案: {greedy_answer}\n")
    print("="*50)
    
    # 2. 运行自洽性方法
    sc_answer, sc_paths, sc_counts = self_consistency_decoding(COT_PROMPT, QUESTION, num_paths=10, model=model)
    
    print("\n--- 自洽性投票结果 ---")
    for answer, count in sc_counts.items():
        print(f"答案 '{answer}' 出现了 {count} 次")
        
    print(f"\n>>> 自洽性解码的最终答案 (最一致): {sc_answer}")

    # 展示最一致答案背后的推理过程
    print(f"\n支持最一致答案 '{sc_answer}' 的其中一条推理路径:")
    print(sc_paths[sc_answer][0])