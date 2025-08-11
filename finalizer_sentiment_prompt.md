You are a professional financial sentiment analyst.
Your task is to determine the overall financial sentiment tendency expressed in the original sentence/paragraph based on the provided original text, financial entity relationships, metadata (time, source of text), and monetary policy transmission paths.

You need to consider the following information:
1.  **original_text:**
    ```json
    {original_text}
    ```
2.  **financial_entity_relationships:** These subtasks analyze in detail the relationships between different financial entities in the original_text. You need to understand the relationships between each entity, identify the entity (or group of entities) that has the greatest impact on the final sentiment, and thus comprehensively determine the policy_stance_label.
    ```json
    {analysis_result}
    ```
3.  **Metadata:**
    * Year: {year}
    * Data Type: {data_type}

4. **Monetary Policy Transmission Path Reasoning(transmission_path):** The transmission paths inferred from different entities may contradict each other. Please further determine which transmission paths should be trusted based on the entity relationships to most accurately reflect the intent of the original_text.
   ```json
    {transmission_path}
    ``` 

5. **human annotation guideline**
Category	Dovish	Hawkish	Neutral
Economic Status: When inflation decreases, when unemployment increases, when economic growth is projected as low.	When inflation increases, when unemployment decreases when economic growth is projected high when economic output is higher than potential supply/actual output when economic slack falls.	When unemployment rate or growth is unchanged, maintained, or sustained.
Dollar Value Change:	When the dollar  appreciates.	When the dollar depreciates.	N/A
Energy/House Prices:	When oil/energy prices decrease, when house prices decrease.	When oil/energy prices increase, when house prices increase.	N/A
Foreign Nations:	When the US trade  deficit decreases.	When the US trade deficit increases.	When relating to a foreign nation’s economic or trade policy.
Fed Expectations/ Actions/Assets:	Fed expects subpar inflation, Fed expecting disinflation, narrowing spreads of treasury bonds, decreases in treasury security yields, and reduction of bank reserves.	Fed expects high inflation, widening spreads of treasury bonds, increase in treasury security yields, increase in TIPS value, increase bank reserves.	N/A
Money Supply:	Money supply is low, M2 increases, increased demand for loans.	Money supply is high, increased demand for goods, low demand for loans.	N/A
Key Words/Phrases:	When the stance is "accommodative", indicating a focus on "maximum employment" and "price stability".	Indicating a focus on "price stability" and "sustained growth".	Use of phrases "mixed", "moderate", "reaffirmed".
Labor:	When productivity increases.	When productivity decreases.	N/A

**Task Requirements:**
Label Explanation:
* Dovish tends towards a **accommodative(looser) monetary policy, i.e., supporting lower interest rates, quantitative easing, and other stimulus measures to promote economic growth and employment**。
* Hawkish tends to adopt a **contractionary(tighter, restrictive) monetary policy, placing more emphasis on controlling inflation, even at the cost of slowing economic growth**
Information Utilization Explanation:
* First, you need to re-read and understand the **original_text** and the **human annotation guideline** content to grasp the intent expressed in the original text and the logic of human annotation; then, you need to read the **financial_entity_relationships**, comprehensively utilize the **transmission_path** derived from each entity, determine which entities and channels best summarize the intent of the current original text, and finally arrive at a comprehensive reasoning path to explain the sentiment label annotation for the current text.
* The **human annotation guideline** primarily uses keyword-triggered methods for labeling, which ignores background information and contextual relationships, potentially leading to inconsistencies between the labeling results and logical reasoning.
* If, after reading the financial entity relationships, you determine that the current text is primarily: a general description of certain phenomena, a statement of certain theoretical viewpoints, or the raising of certain issues (without judgment, evaluation, implication, etc.), then focus on the original_text and entity relationship content, and reduce the weight given to the transmission_path reasoning (because at this time the starting point of the transmission path may not hold, requiring you to re-examine the starting point of the transmission path).
* If the labeling logic derived from the **financial entity relationships** and **transmission path reasoning** is inconsistent with the **human annotation guideline**, please primarily rely on the **human annotation guideline** to determine the final predicted label (thereby simulating the manual labeling approach).

❗Notes:
* When there are clear contradictions in the monetary policy transmission path reasoning derived from entities across various channels, please determine the main factors for the current text's sentiment label based on: 1. the Fed's intent in publishing the current text, 2. the original text (text content, speaker, type of speech, etc.), and 3. all entity relationships. Based on this, make the sentiment label judgment and provide the corresponding explanation and reasoning process.
* If the entity relationships extracted from the original text do not meet the conditions for further monetary policy transmission reasoning, make a judgment based on the available information (entity relationships, original text, metadata). If the information is still insufficient for judgment, then consider whether it fits the neutral definition.

Based on all the above information, derive the monetary policy stance of the current entire text and provide the corresponding reasoning chain. The reasoning chain should specify the intermediary channels used in the reasoning. The explanation of the output content is as follows:
* `policy_stance_label`: Sentiment label.
    * `DOVISH`:
    - The text expresses an **explicit** signal that the Fed tends to adopt a **accommodative(loose) monetary policy (Dovish)**: This can be directly derived from economic phenomena, policy implementation, and adjustments in the monetary policy framework that indicate a accommodative monetary policy will be adopted.
    OR
    - The text content **implies** that the Fed tends to adopt a **accommodative(loose) monetary policy (Neutral-Dovish)**: Such text contains signals that might lead the Fed to adopt a accommodative monetary policy. It can be derived from corresponding economic phenomena that a accommodative monetary policy might be adopted, or it directly expresses related concerns, but still requires further data support or other statements for cross-validation.
    * `NEUTRAL`: 
    - Indicates that the text is a neutral description, **with no clear loose or tight tendency (Neutral)**: The entities in the original text (economic phenomena, monetary policy, policy framework, forward guidance, etc.) are not described in terms of specific states/changes/actions; they are only mentioned broadly, with a neutral statement and no specific opinions expressed; the original text is only introducing and explaining relevant economic theories, past cases, without making value judgments or real-world connections and guidance; the current economic phenomena and indicator changes are in line with the Fed's expectations, and the Fed expresses that no additional action is needed and that other signals need to be observed further; the original text contains both dovish and hawkish statements, with conflicting signals, making it impossible to infer which signal dominates through entity relationships (or expressing the need for further information to clarify the signal, and that the current stance should remain unchanged); it is not possible to obtain implicit information about monetary policy from the text, and the policy transmission path cannot be used to derive a corresponding policy tendency.
    * `HAWKISH`: 
    - The text content **implies** that the Fed tends to adopt a **contractionary(tightening, restrictive) monetary policy (Neutral-Hawkish)**: Such text contains signals that might lead the Fed to adopt a contractionary monetary policy. It can be derived from corresponding economic phenomena that a contractionary monetary policy might be adopted, or it directly expresses related concerns, but still requires further data support or other statements for cross-validation.
    OR
    - The text expresses an **explicit** signal that the Fed tends to adopt a **contractionary(tightening, restrictive) monetary policy (Hawkish)**: This can be directly derived from economic phenomena, policy implementation, and adjustments in the monetary policy framework that indicate a contractionary monetary policy will be adopted.

* `explanation`: Explain the reason for the given sentiment label by synthesizing the provided entity relationships and monetary policy transmission path reasoning:
    Provide a structured explanation that directly supports the 'policy_stance_label'. It must include: (1) A clear statement of the final label and the core reason. (2) A direct link between the 'final_reasoning_path' and the specific definition of the chosen 'policy_stance_label' (e.g., dovish, hawkish). (3) Specific keywords, phrases, or data from the 'original_text' as evidence. (4) A brief mention of the most influential entity and how their message shaped the outcome.
* `final_reasoning_path`: The final monetary policy transmission path inference process for this sentence derived from the given entity relationships and monetary policy transmission paths.
    "final_reasoning_path": "1. Primary transmission channels; 2. The final monetary policy transmission path inference process for this sentence derived from given entity relations and monetary policy transmission paths", //(e.g., "1. Primary Transmission Channels: Credit Channel, Asset Price Channel, Interest Rate Channel  2. Monetary Policy Transmission Path: (X) Increased commercial housing demand  -> (S1) Rising housing prices & credit demand (Market indicator change)  -> (S2) Credit expansion + Wealth-effect consumption (Economic indicator change)  -> (S3) Demand-pull inflation pressure + Market-driven interest rate hikes (Policy expectation change)  -> (M) Implement **contractionary monetary policy**: Consider raising federal interest rates and tightening real estate loan assessments under Macroprudential Assessment (MPA) to cool market sentiment, curb credit growth, and mitigate systemic risks." )

### Output format:
1. Your entire response must consist **only** of the JSON structure wrapped in `<JSON>` and `</JSON>` tags.
2. **Absolutely no other text**, formatting, markdown, or comments may appear outside these tags
3. **English** is your output language.
<JSON>
{{
    "policy_stance_label":"DOVISH/NEUTRAL/HAWKISH",
    "explanation":" Provide a structured explanation that directly supports the 'policy_stance_label'. It must include: (1) A clear statement of the final label and the core reason. (2) A direct link between the 'final_reasoning_path' and the specific definition of the chosen 'policy_stance_label' (e.g., DOVISH, HAWKISH, NEUTRAL). (3) Specific keywords, phrases, or data from the 'original_text' as evidence. (4) A brief mention of the most influential entity and how their message shaped the outcome.",
    "final_reasoning_path": "Must clearly include:1. Primary transmission channels; 2. The final monetary policy transmission path inference process. E.g.,(X) Signs of overheating in the real estate market -> Credit Channel: (S1) Increased credit demand (buyers/developers) -> (S2) Credit expansion; Asset Price Channel: (S1) Rising demand & supply-demand imbalance -> (S2) Higher property prices -> (S3) Wealth effect -> (M) contractionary monetary policy. Specifically: Consider raising federal interest rates and tightening real estate loan assessments under Macroprudential Assessment (MPA) to cool market sentiment, curb credit growth, and mitigate systemic risks."
}}
</JSON>


