# Deep Research Source and Discourse Analysis

This repository contains an analysis comparing the Deep Research writing process with human-written articles from Wikipedia and established news sources. The goal is to explore how these sources differ in their citation practices and narrative structures.

We label the function, authority, and perspective of every citation, and map the discourse role of each sentence. This analysis reveals the fundamental building blocks of different writing styles.

## Key Research Questions
*   How does the sourcing strategy of Deep Research differ from that of journalists and Wikipedia contributors?
*   Are there measurable differences in how these sources construct narratives and structure their arguments?
*   What patterns emerge in terms of source authority, narrative function, and discourse flow?

## File Structure

The key files and directories for this project are organized as follows:

*   `data/`: This directory contains all raw, intermediate, and final datasets used in the analysis. This includes the parsed articles, citation summaries, and the final labeled dataframes for both citation and discourse analysis.

*   `notebooks/`: This directory contains the Jupyter notebooks used for data processing and analysis.
    *   `notebooks/2025-06-09__label-sources.ipynb`: This is the **main analysis notebook**. It contains all the primary visualizations and statistical comparisons across the three sources (Deep Research, News, Wikipedia).
    *   **Auxiliary Notebooks**: Other notebooks in this directory are used for data cleaning, parsing, and preparation steps (e.g., summarizing citations, structuring discourse data) before the final analysis.
    *   **utils_helper.py**: This file contains the helper functions for the source analysis and PROMPTs.
    *   **utils_discourse.py**: This file contains the helper functions for the discourse analysis and PROMPTs.

## Key Insights

The analysis has revealed several significant differences in writing style:

### Sourcing Strategy

![Source Authority](figures/centrality-heatmap-by-index.png)

1.  **Sourcing Strategy**: News articles prioritize expert opinions and authoritative figures early in the text, while Deep Research tends to rely more heavily on journalistic and summary-style sources.
    *   **First Source as Expert**: **59%** of initial sources in news articles are from an "Individual/Expert," compared to **0%** in Deep Research.
    *   **Overall Source Mix**: Deep Research uses "Journalistic" sources for **38%** of its citations, compared to **18%** in news articles.

![Source Usage](figures/narrative_function_ratios.png)

2.  **Argument Construction**: The analysis reveals a clear "inverted authority pyramid" in news articles. News articles begin with high-impact, central sources, whereas Deep Research builds its case from broader, more contextual sources.
    *   **Centrality of First Source**: In news articles, **64%** of initial citations are of "High" centrality to the core argument. In Deep Research, this figure is only **20%**.
    *   **Use of Contextual Sources**: Conversely, **80%** of the initial sources in Deep Research are of "Low" centrality, compared to just **14%** in news.

### Discourse Structure

![Discourse Structure](figures/discourse_structure.png)

3.  **Narrative Structure**: Deep Research articles dedicate more space to the **implications** and **broader context** of an issue, while news articles focus more on presenting **counterpoints** and direct **evidence**.
    *   **Implications**: **26%** of sentences in Deep Research articles discuss implications, versus only **13%** in news.
    *   **Counterpoints**: News articles use citations to introduce a "Counterpoint" **13%** of the time, compared to **8%** in Deep Research.

4.  **Narrative Function**: Deep Research articles arrive at their **main idea** later in the text, while news articles build their argument from the beginning.
    *   **Main Idea**: Deep Research articles arrive at their **main idea** later in the text, while news articles build their argument from the beginning.
    *   **History and Background**: Deep Research and Wikipedia articles both use citations to provide historical context initially.


For more insights, see [this Google Docs document](https://docs.google.com/document/d/1kblZv_SPD3BSvutNMrJpH3vCk3YOZS7DMjd3bw3fmIA/edit?usp=sharing).

For a detailed breakdown of the analytical categories and labels used, please refer to `notebooks/utils_helper.py` (for citation analysis) and `notebooks/utils_discourse.py` (for discourse analysis).

Example of a citation and how we processed it:

```{
    "source": "Deep Research",
    "citation": "3 Minutes on Panda Diplomacy | American Magazine | American ...",
    "Information": "Pandas, under President Xi Jinping's leadership, are considered as 'envoys of friendship' in China's strategic endeavors. These strategic loans are used to advance China's national objectives, enhance global influence, and spread its rich cultural history. Panda loans often coincide with significant diplomatic events or anniversaries, underscoring their role in diplomatic relations. This strategic approach also aligns with creating favorable public opinions internationally, as the 'envoy of friendship' narrative promotes warmth and generous perceptions, balancing the strategic realities and financial commitments involved in these loan agreements. This strengthens China's diplomatic arsenal, elevating pandas beyond cultural symbols to strategic instruments in global diplomacy.",
    "date_accessed": "June 6, 2025",
    "Narrative_Function": "Supporting Evidence",
    "Perspective": [
        "Supportive"
    ],
    "Centrality": "Medium",
    "Temporal_Role": "Current event",
    "Citation_Type": "News article",
    "Scope": "International/comparative",
    "Source_Authority": "Media organization",
}
```

## Future Work

This analysis is ongoing.