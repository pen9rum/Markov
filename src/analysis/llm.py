"""
LLM模块 - 用于分析玩家行为 (支持多種雲端API)

支持的模型：
- Qwen API: qwen-plus, qwen-turbo, qwen-max-latest
- Gemini API: gemini-3-flash-preview (Gemini 3 Flash), gemini-3.1-pro-preview (Gemini 3.1 Pro)
- OpenAI API: gpt-5-mini, gpt-5
- OpenAI-compatible 本地端点: Falcon-H1（可CPU运行）
"""
from typing import Tuple, Dict, Any
import os
import requests
import json


# 自动加载 .env 文件
def load_env_file():
    """从 .env 文件加载环境变量"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # 使用 .env 中的值覆盖旧环境变量，避免误用过期/错误 key
                    os.environ[key] = value

# 启动时自动加载
load_env_file()


def get_response(prompt: str,
                 model_name: str = "qwen-plus",
                 api_key: str = None,
                 api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 **kwargs) -> Tuple[dict, str]:
    """
    调用Qwen API获取响应
    
    Args:
        prompt: 输入提示词
        model_name: 模型名称，建议使用:
                   - "qwen-plus" (推荐，平衡性能)
                   - "qwen-max-latest" (最新最强)
                   - "qwen-turbo" (快速便宜)
        api_key: API密钥，如果为None则从环境变量DASHSCOPE_API_KEY读取
        api_url: API端点
        **kwargs: 其他参数 (temperature, max_tokens等)
    
    Returns:
        (响应字典, 输出文本)
    """
    # 获取API密钥
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set DASHSCOPE_API_KEY environment variable or pass api_key parameter.")
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **kwargs
    }
    
    # 发送请求
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    output_text = result['choices'][0]['message']['content']
    
    return result, output_text


def get_response_gemini(prompt: str,
                        model_name: str = "gemini-3-flash-preview",
                        api_key: str = None,
                        temperature: float = 1.0,
                        max_tokens: int = 8192,
                        **kwargs) -> Tuple[dict, str]:
    """
    调用Gemini API获取响应
    
    Args:
        prompt: 输入提示词
        model_name: 模型名称，默认 gemini-3-flash-preview
        api_key: API密钥，如果为None则从环境变量GEMINI_API_KEY读取
        temperature: 温度参数 (0-2)
        max_tokens: 最大生成token数
        **kwargs: 其他参数
    
    Returns:
        (响应字典, 输出文本)
    """
    # 获取API密钥
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # Gemini API endpoint
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
    # 构建请求
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {
                "thinkingLevel": "low"
            },
            **kwargs
        }
    }
    
    # 发送请求（API key 作为查询参数）
    params = {"key": api_key}
    response = requests.post(api_url, headers=headers, json=payload, params=params)
    
    # 检查响应状态
    if response.status_code != 200:
        error_detail = response.json() if response.text else {"error": "Unknown error"}
        raise Exception(f"Gemini API error: {response.status_code} - {error_detail}")
    
    result = response.json()
    
    # 提取文本输出
    try:
        output_text = result['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError) as e:
        raise Exception(f"Failed to parse Gemini response: {e}\nResponse: {result}")
    
    # 构建元数据
    metadata = {
        "model": model_name,
        "usage": result.get('usageMetadata', {}),
    }
    
    return result, output_text

def get_response_openai(prompt: str,
                        model_name: str = "gpt-5-mini",
                        api_key: str = None,
                        api_url: str = "https://api.openai.com/v1/responses",
                        max_tokens: int = 8192,
                        reasoning_effort: str = "low",
                        verbosity: str = "low",
                        **kwargs) -> Tuple[dict, str]:
    """
    调用 OpenAI Responses API 获取响应

    Args:
        prompt: 输入提示词
        model_name: OpenAI 模型名称，例如 gpt-5-mini 或 gpt-5
        api_url: Responses API 端点（默认官方 OpenAI）
    """
    is_official_openai = "api.openai.com" in api_url.lower()

    if api_key is None:
        if is_official_openai:
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            # 本地 OpenAI-compatible 服务通常不强制 API key
            api_key = os.environ.get("FALCON_LOCAL_API_KEY", "")

    if is_official_openai and not api_key:
        raise ValueError(
            "API key not found. Please set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "reasoning": {
            "effort": reasoning_effort
        },
        "text": {
            "verbosity": verbosity
        }
    }
    payload.update(kwargs)

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        raise Exception(f"OpenAI API error: {response.status_code} - {error_detail}")

    result = response.json()

    # 先嘗試頂層 convenience 欄位
    output_text = result.get("output_text", "")

    # 如果沒有，就手動從 output 陣列裡抽文字
    if not output_text:
        texts = []
        for item in result.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
        output_text = "\n".join(t for t in texts if t)

    # 真的還是空，再印完整 raw response
    if not output_text:
        print("OpenAI raw response:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result, output_text


def get_response_openai_compatible_local(prompt: str,
                                                                                 model_name: str = "falcon-h1-local",
                                                                                 max_tokens: int = 8192,
                                                                                 **kwargs) -> Tuple[dict, str]:
        """
        调用本地 OpenAI-compatible Responses API（默认用于 Falcon-H1）。

        通过环境变量配置：
            - FALCON_LOCAL_API_URL: 例如 http://127.0.0.1:8080/v1/responses
            - FALCON_LOCAL_MODEL:   例如 tiiuae/Falcon-H1-7B-Instruct
            - FALCON_LOCAL_API_KEY: 可选，若本地服务要求鉴权则设置
        """
        api_url = os.environ.get("FALCON_LOCAL_API_URL", "http://127.0.0.1:8080/v1/responses")
        requested_model = model_name

        if model_name in {"falcon-h1-local", "falcon-h1"}:
                requested_model = os.environ.get("FALCON_LOCAL_MODEL", "tiiuae/Falcon-H1-7B-Instruct")

        return get_response_openai(
                prompt=prompt,
                model_name=requested_model,
                api_url=api_url,
                max_tokens=max_tokens,
                reasoning_effort="low",
                verbosity="low",
                **kwargs
        )

def get_response_deepseek(prompt: str,
                          model_name: str = "deepseek-chat",
                          api_key: str = None,
                          max_tokens: int = None,
                          **kwargs) -> Tuple[dict, str]:
    """
    调用DeepSeek API获取响应
    
    Args:
        prompt: 输入提示词
        model_name: 模型名称，默认 deepseek-chat (支持 deepseek-chat, deepseek-reasoner)
        api_key: API密钥，如果为None则从环境变量DEEPSEEK_API_KEY读取
        max_tokens: 最大生成token数 (deepseek-reasoner默认32768, deepseek-chat默认8192)
        **kwargs: 其他参数
    
    Returns:
        (响应字典, 输出文本)
    """
    # 获取API密钥
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # 根据模型设置默认 max_tokens
    if max_tokens is None:
        if "reasoner" in model_name.lower():
            max_tokens = 65536  # reasoner 在長對局中思考消耗大量 token，需要更大上限
        else:
            max_tokens = 8192   # chat 默认值
    
    # DeepSeek API endpoint (OpenAI-compatible)
    api_url = "https://api.deepseek.com/v1/chat/completions"
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        **kwargs
    }
    
    # 发送请求，增加超时时间（reasoner 需要更长时间）
    timeout = 300 if "reasoner" in model_name.lower() else 120
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    
    # 检查响应状态
    if response.status_code != 200:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        raise Exception(f"DeepSeek API error: {response.status_code} - {error_detail}")
    
    result = response.json()
    
    # 检查是否因为长度限制而截断
    finish_reason = result.get("choices", [{}])[0].get("finish_reason", "")
    if finish_reason == "length":
        print(f"⚠️ 警告: DeepSeek 响应因达到 max_tokens={max_tokens} 而被截断！")
        print(f"建议增加 max_tokens 参数或简化 prompt")
    
    # 提取文本输出
    try:
        message = result["choices"][0]["message"]
        reasoning_content = message.get("reasoning_content", "")
        content = message.get("content", "")
        
        # debug 用 - 如果有 reasoning_content，顯示前 1000 字
        if reasoning_content:
            print("DeepSeek reasoning_content (前1000字):")
            print(reasoning_content[:1000])
        
        # 如果 content 是空的，印出完整回應
        if not content:
            print("DeepSeek raw response:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 優先使用 content，如果沒有則用 reasoning_content
        output_text = content or reasoning_content
        
    except (KeyError, IndexError) as e:
        raise Exception(f"Failed to parse DeepSeek response: {e}\nResponse: {result}")
    
    return result, output_text


def get_player_knowledge_base() -> str:
    """
    获取所有玩家的先导知识 - 行为特征描述
    """
    knowledge = """
# Rock-Paper-Scissors Player Behavior Knowledge Base

## Distribution Strategy Players
These players always choose the same action:

**A - Pure Scissors**: Always plays Scissors (0% Rock, 0% Paper, 100% Scissors)
- Completely predictable
- Will lose to Rock players, beat Paper players, tie with Scissors players

**B - Pure Rock**: Always plays Rock (100% Rock, 0% Paper, 0% Scissors)
- Completely predictable
- Will beat Scissors players, lose to Paper players, tie with Rock players

**C - Pure Paper**: Always plays Paper (0% Rock, 100% Paper, 0% Scissors)
- Completely predictable
- Will beat Rock players, lose to Scissors players, tie with Paper players

**D - Uniform Random**: Equal probability for all actions (33.3% Rock, 33.3% Paper, 33.4% Scissors)
- Unpredictable order but balanced distribution
- No exploitable pattern

**E - Rock + Paper**: Only plays Rock or Paper (50% Rock, 50% Paper, 0% Scissors)
- Never plays Scissors
- Vulnerable to Scissors-heavy strategies

**F - Rock + Scissors**: Only plays Rock or Scissors (50% Rock, 0% Paper, 50% Scissors)
- Never plays Paper
- Vulnerable to Paper-heavy strategies

**G - Paper + Scissors**: Only plays Paper or Scissors (0% Rock, 50% Paper, 50% Scissors)
- Never plays Rock
- Vulnerable to Rock-heavy strategies

**H - Rock Biased**: Prefers Rock (50% Rock, 25% Paper, 25% Scissors)
- Rock appears twice as often as other moves
- Can be exploited by Paper-biased strategies

**I - Paper Biased**: Prefers Paper (25% Rock, 50% Paper, 25% Scissors)
- Paper appears twice as often as other moves
- Can be exploited by Scissors-biased strategies

**J - Scissors Biased**: Prefers Scissors (25% Rock, 25% Paper, 50% Scissors)
- Scissors appears twice as often as other moves
- Can be exploited by Rock-biased strategies

**K - Rock > Paper**: Strong Rock preference (50% Rock, 33.3% Paper, 16.7% Scissors)
- Rock most common, Paper second, Scissors least
- Hierarchy: Rock > Paper > Scissors

**L - Rock > Scissors**: Strong Rock preference (50% Rock, 16.7% Paper, 33.3% Scissors)
- Rock most common, Scissors second, Paper least
- Hierarchy: Rock > Scissors > Paper

**M - Paper > Rock**: Strong Paper preference (33.3% Rock, 50% Paper, 16.7% Scissors)
- Paper most common, Rock second, Scissors least
- Hierarchy: Paper > Rock > Scissors

**N - Paper > Scissors**: Strong Paper preference (16.7% Rock, 50% Paper, 33.3% Scissors)
- Paper most common, Scissors second, Rock least
- Hierarchy: Paper > Scissors > Rock

**O - Scissors > Rock**: Strong Scissors preference (33.3% Rock, 16.7% Paper, 50% Scissors)
- Scissors most common, Rock second, Paper least
- Hierarchy: Scissors > Rock > Paper

**P - Scissors > Paper**: Strong Scissors preference (16.7% Rock, 33.3% Paper, 50% Scissors)
- Scissors most common, Paper second, Rock least
- Hierarchy: Scissors > Paper > Rock

## Markov Based Players 
These players adapt their strategy based on opponent's previous moves:

**X - Win-Last**: Plays the move that would have beaten opponent's last move
- First move: Random
- Subsequent moves: Counter opponent's previous move
- Against Pure Rock (always Rock): Will play Paper after first round
- Highly effective against predictable opponents

**Y - Lose-Last**: Plays the move that would have lost to opponent's last move
- First move: Random
- Subsequent moves: Intentionally lose to opponent's previous move
- Against Pure Rock: Will keep playing Scissors
- Intentionally losing strategy

**Z - Copy-Last**: Copies opponent's last move
- First move: Random
- Subsequent moves: Mirrors opponent's previous move
- Against Pure Rock: Will always play Rock (all ties)
- Creates mirror matches

## Key Insights:
1. Pure strategies are completely predictable and can be easily exploited
2. Distribution strategies maintain consistent ratios but unpredictable order
3. Reactive strategies are most effective against predictable opponents
4. Win-Last (X) tends to perform best overall
5. Strategies with heavy bias can be counter-strategized
"""
    return knowledge


def analyze_game_trajectory(player1_id: str, player2_id: str, 
                            player1_trajectory: str, player2_trajectory: str,
                            player1_wins: int, player2_wins: int, draws: int,
                            num_rounds: int,
                            api_type: str = "qwen",
                            model_name: str = None) -> Dict[str, Any]:
    """
    使用雲端LLM API分析游戏轨迹
    
    Args:
        player1_id: 玩家1 ID
        player2_id: 玩家2 ID
        player1_trajectory: 玩家1轨迹
        player2_trajectory: 玩家2轨迹
        player1_wins: 玩家1胜场
        player2_wins: 玩家2胜场
        draws: 平局数
        num_rounds: 总回合数
        api_type: API类型，支持 qwen/gemini/openai/deepseek/openai_compat_local
        model_name: 模型名称，如果为None则使用默认值
    
    Returns:
        包含分析结果的字典
    """
    knowledge_base = get_player_knowledge_base()
    
    prompt = f"""{knowledge_base}

## Game Analysis Task

You are a Rock-Paper-Scissors strategy expert. Above is the background knowledge of 19 player strategies.

Now analyze a game where both players' identities are unknown based on their move trajectories:

**CRITICAL CONSTRAINTS**:
- Valid player identities are ONLY: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P (distribution strategies) and X, Y, Z (Markov/reactive strategies)
- You MUST only use these 19 letters. Do NOT invent other letters like U, T, etc.
- Your final answer MUST contain both lines: "Player1: ..." and "Player2: ..." and they cannot be empty.
- Even if uncertain, you MUST provide your single best guess (no abstain / no refusal).
- You MUST provide concrete numeric counts for Rock, Paper, Scissors for BOTH players.
- Forbidden placeholders in final answer: "[count ...]", "cannot be computed", "same", "N/A", "unknown", "TBD".
At most, only one of the players will be Markov player (X, Y, Z). The other player will be from A-P. It is also possible that both players are from A-P.
The Markov player's strategy comes from last player. Therefore, you have to analyze the trajectory every round to determine which player is Markov and which is distribution. If both players are from A-P, then you can analyze them as two distribution players.
The best solution is to follow the following steps and think step by step:
(1) Find out is there any player is Markov player (X, Y, Z) by analyzing the trajectory round by round. If there is a Markov player, then you can determine the other player's identity as well.
(2) If there is no Markov player, then you can analyze the trajectory as two distribution players and find out the best matching identities for both players.
(3) Based on the identified player identities, count how many times rock, paper, and scissors appear in each player's {num_rounds}-round trajectory.
**Game Info**:
- Total Rounds: {num_rounds}
- Results: Player1 won {player1_wins}, Player2 won {player2_wins}, Draws {draws}

**Player1 Trajectory**:
{player1_trajectory}

**Player2 Trajectory**:
{player2_trajectory}
Please respond with a detailed analysis of the players' strategies, their most likely identities, and the predicted probabilities for their next moves. Be sure to justify your reasoning based on the trajectories and the knowledge base provided.
After your detailed analysis, start your final answer with "Final Answer:" and provide the identified player identities and the predicted probabilities in a clear format with:
Player1: Identity, Rock count, Paper count, Scissors count
Player2: Identity, Rock count, Paper count, Scissors count
where count means the actual number of times the player played that action in the whole trajectory, which can be used to verify the correctness of your analysis.
Use EXACTLY this output template at the end:
Final Answer:
Player1: <Identity>, Rock count=<int>, Paper count=<int>, Scissors count=<int>
Player2: <Identity>, Rock count=<int>, Paper count=<int>, Scissors count=<int>
"""
    
    print(f"\n{'='*80}")
    print(f"Sending request to {api_type.upper()} API for analysis...")
    print(f"{'='*80}\n")
    
    try:
        if api_type.lower() == "gemini":
            # 使用 Gemini API
            if model_name is None:
                model_name = "gemini-3-flash-preview"
            response, output_text = get_response_gemini(prompt, model_name=model_name)
            metadata = {
                "model": model_name,
                "usage": response.get('usageMetadata', {}),
            }
        elif api_type.lower() == "openai":
            # 使用 OpenAI API
            if model_name is None:
                model_name = "gpt-5-mini"
            response, output_text = get_response_openai(prompt, model_name=model_name)
            metadata = {
                "model": response.get("model"),
                "usage": response.get("usage"),
            }
        elif api_type.lower() == "deepseek":
            # 使用 DeepSeek API
            if model_name is None:
                model_name = "deepseek-chat"
            response, output_text = get_response_deepseek(prompt, model_name=model_name)
            metadata = {
                "model": response.get("model"),
                "usage": response.get("usage"),
            }
        elif api_type.lower() == "openai_compat_local":
            # 使用本地 OpenAI-compatible API（Falcon-H1）
            if model_name is None:
                model_name = "falcon-h1-local"
            response, output_text = get_response_openai_compatible_local(prompt, model_name=model_name)
            metadata = {
                "model": response.get("model", model_name),
                "usage": response.get("usage"),
                "endpoint": os.environ.get("FALCON_LOCAL_API_URL", "http://127.0.0.1:8080/v1/responses")
            }
        else:
            # 使用 Qwen API (默认)
            if model_name is None:
                model_name = "qwen-plus"
            response, output_text = get_response(prompt, model_name=model_name)
            metadata = {
                "model": response.get("model"),
                "usage": response.get("usage"),
            }
        
        return {
            "success": True,
            "player1_id": player1_id,
            "player2_id": player2_id,
            "analysis": output_text,
            "response_metadata": metadata
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "player1_id": player1_id,
            "player2_id": player2_id,
        }
