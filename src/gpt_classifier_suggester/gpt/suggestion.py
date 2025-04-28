import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env环境变量
load_dotenv()

# 从环境变量中读取 OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "❌ OPENAI_API_KEY not found in environment variables. Please set it in your .env file."
    )

# 初始化 OpenAI 客户端
client = OpenAI(api_key=api_key)


def generate_gpt_suggestions(text, pet_type, prob, model_name="gpt-4"):
    prompt = f"""
You are a Reddit content optimization expert.

The following is a Reddit adoption post for a **{pet_type}**. Our predictive model estimates that this post is likely to have **{'low' if prob < 0.5 else 'high'} engagement** (predicted probability: {prob:.2f}).

Please provide **three clear and specific suggestions** to improve the language or structure of the post, focusing on increasing user engagement (likes, comments, shares).

You can consider:
- Emotional tone and word choice
- Clarity and specificity of the pet’s description
- Use of urgency or calls to action
- Community-oriented language

📝 Original post:
{text}

Format your response as a numbered list with concise English suggestions (1–2 sentences each).
Only return the suggestions.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ GPT suggestion failed: {str(e)}"
