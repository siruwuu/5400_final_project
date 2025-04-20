# q1
## Posts
模型 | RMSE | R²
🐱 Cats | 4527.75 | 0.006
🐶 Dogs | 991.09 | 0.005

R² 仍然很低：说明语言风格只能解释互动的极小部分，但我们仍然能从特征系数中获得有用洞察。

### 高影响特征（正相关）
特征                    | 🐱 Cats Coef | 🐶 Dogs Coef | 说明
contains_adopt_keywords | +656        | +85         | 使用“adopt”等关键词显著提升互动，猫帖效果更强
num_exclamations        | +77         | +43         | 强调语气有助于互动
title_length            | +11         | +2          | 较长标题稍有帮助
sentiment_score         | ❌-157      | ✅+97        | 猫帖情绪越正越没反应，狗帖正向情绪反而略微正面

猫帖里如果情绪太正面（如：“She's perfect, sweet, cuddly and ready!”）可能会被用户识别为“套路”或“过度美化”，反而降低了可信度。
而狗帖中使用正向语言（如 “friendly,” “loyal,” “playful”）通常更容易被接受，因为狗在语境中本来就被视为“忠诚、友善”，这些描述可能更自然、更具说服力。

### 高影响特征（负相关）
特征              |CatsCoef|DogsCoef| 说明
has_pronouns      | -2377 | -804 | 使用 “you”, “we” 的贴子反而更不受欢迎（可能被认为是推销/不真实）
has_question      | -1350 | -323 | 提问语气（如“anyone interested?”）反而降低互动率
contains_money    | -1230 | -247 | 涉及捐款/资金的贴子互动低，可能因商业味道过重
has_urgency_words | -912  | -309 | 使用紧急语言如“urgent”未必能提高互动，甚至可能被认为“套路”

### Post_Report
We conducted a linear regression analysis to evaluate the impact of various linguistic features on post engagement, measured by a composite metric combining upvotes and comment counts. The results suggest that the presence of adoption-related keywords (e.g., “adopt,” “rescue,” “rehome”) is the strongest positive predictor of engagement in both cat and dog posts, with a more pronounced effect in cat posts.

Interestingly, several features typically considered persuasive—such as personal pronouns (“you,” “we”), urgency words (“urgent,” “help,” “last chance”), and mentions of money (“donation,” “fund,” “$”)—were all negatively associated with engagement. This may reflect a form of audience skepticism on Reddit, where users are less responsive to emotionally charged or overtly promotional language.

Additionally, question forms (e.g., “Can anyone help?” or “Interested?”) had a significant negative correlation with engagement, suggesting that indirect or vague calls-to-action may reduce post effectiveness. On the other hand, posts with emphasis markers like exclamation points and slightly longer titles tended to receive more engagement, albeit modestly.

Overall, these findings indicate that authentic, informative, and action-oriented language—particularly posts that clearly reference adoption—are more effective than emotionally loaded appeals. The divergent impacts observed between cat and dog posts also suggest that audience expectations may vary by pet type, which could be an area for further analysis.



## Comments

特征名                   | Cats 评论系数 | Dogs 评论系数 | 说明
contains_adopt_keywords | +30.77       | +8.40 | 提及“adopt”、“rescue”等关键词能显著提升评论点赞，猫帖评论中影响力更大
has_pronouns            | +12.03       | +3.31 | 人称代词“you”、“we”有正面影响，表明亲和式语言能提升评论共鸣，和原帖中的负面作用形成对比
sentiment_score         | +4.25        | ❌-9.54 | 在猫帖评论中正向情绪略有提升互动，但狗帖评论中表现为负相关，情绪太浓反而略微降低点赞
num_adjectives          | +0.60        | +0.76 | 描述性语言（如“cute”, “friendly”）在两个物种中影响均较小但为正
num_verbs               | +0.47        | -0.26 | 动词使用量对互动影响较弱，猫帖为正，狗帖为负（可能因命令句过多？）
num_words               | -0.22        | +0.05 | 评论字数对互动影响极小，可能不是决定性因素
num_exclamations        | -0.18        | -0.57 | 感叹号未提升互动，甚至略有负面（可能被视为不够理性）
num_emojis              | -0.48        | -1.03 | emoji 表情对评论表现没有帮助，在狗帖中负面更强
has_question            | -16.54       | ≈0 | 猫帖评论中问句显著降低点赞，狗帖中影响几乎为零
has_urgency_words       | -30.64       | -7.62 | “urgent”、“please”等紧迫性语言仍然为负面，可能触发读者抵触心理

### Report
We analyzed over 65,000 comments across cat and dog adoption posts to understand which linguistic features are most strongly associated with upvote counts. In both datasets, the presence of adoption-related keywords (e.g., “adopt,” “rescue”) was the top predictor of higher comment engagement—indicating that comments that reinforce the adoption theme are more likely to be valued by the community.

Interestingly, we observed that personal pronouns such as “you” and “we” were positively associated with comment popularity in cat and dog threads alike, suggesting that more direct or inclusive language may enhance resonance and connection with readers—contrasting with their negative impact in original posts. In cat comments, positive sentiment scores also predicted more upvotes, but this effect reversed in dog comments, where more emotional or “sweet” language showed a slight negative relationship with engagement.

Certain stylistic elements such as question marks, exclamation points, and emoji had little to no positive influence, and in some cases (especially for cats), actually reduced engagement. Notably, the use of urgency-related words (e.g., “urgent,” “please,” “last chance”) again showed a strong negative association, aligning with our earlier finding that emotionally manipulative language may trigger user skepticism on Reddit.

Despite the low R² values (less than 1% of engagement variance explained), the consistent directionality of coefficients across both species supports a key insight: comments that are thematically aligned, moderately expressive, and personally inclusive tend to perform better, while those perceived as overly emotional or formulaic may be dismissed by the community.

## Compare between Posts and Comments
特征名 | 🐱 Cats Post Coef | 🐶 Dogs Post Coef | 🐱 Cats Comment Coef | 🐶 Dogs Comment Coef | 说明
contains_adopt_keywords | +656 | +85 | +30.8 | +8.4 | 最稳定的正向特征——强调“adopt”、“rescue”都能提升互动，无论是帖子还是评论
has_pronouns | -2377 | -804 | +12.0 | +3.3 | 逆转特征：在帖子中，人称代词大幅降低互动；在评论中却是轻微正向，说明评论更适合亲和型语气
sentiment_score | -157 | +97 | +4.25 | -9.54 | 情绪语言对评论影响更温和，在狗帖中依然负向，猫帖中略为正向，呈现复杂差异
num_exclamations | +77.7 | +42.9 | -0.18 | -0.57 | 在帖子中强调语气有帮助，但在评论中反而降低可信度或被认为是“情绪用力过猛”
num_emojis | -53.1 | -16.4 | -0.48 | -1.03 | emoji 一致为负面：无论在哪个层级都不提升互动，可能因显得不够真实或显得不专业
has_question | -1350 | -323 | -16.5 | ≈0 | 问句在帖子中有显著负面影响，在评论中几乎无影响，说明问句对首发贴损害最大
has_urgency_words | -912 | -309 | -30.6 | -7.6 | 一致性强的负向特征——“紧急感”语言被 Reddit 用户普遍不喜欢，认为“套路感强”

By comparing the linguistic impact across both post and comment levels, we found that some features—such as adoption-related keywords—consistently drive engagement, while others (e.g., urgency words, emojis, and question formats) are negatively associated in nearly all contexts. Interestingly, personal pronouns and sentiment have diverging effects: they lower engagement in posts but show modest positive or neutral impact in comments. This suggests that Reddit users may value clarity and authenticity in original posts, while favoring relatability and conversational tone in comment interactions.