import os 
from openai import OpenAI
from pydantic import BaseModel
import json
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

RUN_SOURCE_LABELING = False


os.environ['OPENAI_API_KEY'] = open(os.path.expanduser('~/.openai-bloomberg-project-key.txt')).read().strip()
client = OpenAI()

def format_citation(c):
    output = []
    if 'Name' in c:
        output.append('Name: ' + c['Name'])
    if 'citation_id' in c:
        output.append('Citation ID: ' + c['citation_id'])
    if 'Biography' in c:
        output.append('Details: ' + c['Biography'])
    if 'Information' in c:
        output.append('Information provided: ' + c['Information'])
    if 'citation' in c:
        output.append('Citation: ' + c['citation'])
    return '\n'.join(output)

def prompt_openai(prompt: str, response_format: BaseModel = None):
    if response_format:
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=response_format
        )    
        return response.choices[0].message.parsed
    else:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    


SUMMARY_PROMPT = """
Please summarize this article in 1-2 sentences. Return just the summary, no other text.

<article_title>
{article_title}
</article_title>

<article>
{article}
</article>

Your response:
"""

ANALYSIS_PROMPT = """
You will receive a summary of an article and a set of {k} citations, along with the information they contain.

Examine the content of the citation and what they provide. Don't get too caught up in the details of where the citation came from (e.g. whether it is an article, a person, etc.)

For each citation in the list, provide the following attributes once per citation:

(1) ID: Copy the name or ID of the citation.

(2) Narrative_Function: Provide a generic keyword label categorizing the narrative role of the citation in the article. Infer why the author used the citation, and give a generalizable description of its narrative function. Don't simply summarize their identity. Return in the format: "LABEL": DESCRIPTION.
Choose one of the following labels:
- "Historical Background": Provides historical context to events being described in the article.
- "Explanation": Explains details or rationale behind policies or actions.
- "Counterpoint": Offers alternative or contrasting views to the main narrative.
- "Supporting Evidence": Strengthens claims made by the article with empirical or expert backing.
- "Demonstration of Impact": Highlights the real-world consequences or significance of described events.
- "Future Projection": Offers forecasts, predictions, or future implications of current events.
- "Expert Commentary": Presents analysis or interpretation from experts.
- "Human Perspective": Gives a personal or community perspective, grounding abstract points in lived experience.
- "Color": Humor or non-factual content that helps with the tone and readability of the article.
- "Additional Context": Expands understanding by offering broader context or setting.
- "Summarizing": Presents or summarizes other ideas and information in the article.
- "Transitional": Provides a transition between ideas or sections of the article.
- "Other": Does not fit into predefined categories.
- "Cannot Determine": Insufficient information to categorize.


(3) Perspective: Identify the citation's perspective on the main events described. Consider the tone, stance, and intention behind the citation regarding the main narrative.
Select one or more of the following labels (do not select contradicting labels):
- "Authoritative": Highly credible, trusted information often from official sources or expert consensus.
- "Informative": Primarily presents factual information without taking a strong stance.
- "Supportive": Clearly aligns with or reinforces the main narrative of the article.
- "Skeptical": Expresses doubts or raises questions about the main narrative.
- "Against": Directly opposes or contradicts the main narrative.
- "Neutral": Neither supports nor opposes the narrative explicitly.
- "Cannot Determine": Insufficient information to categorize.

(4) Centrality: Evaluate how crucial the citation is to the main events of the article. Assess based on the extent to which removing this citation would significantly alter the understanding or completeness of the narrative.
Select one label from:
- "High": Essential to the narrative; removing it significantly weakens or alters the main points.
- "Medium": Provides important context or support but is not indispensable.
- "Low": Offers minor or supplemental information; its removal does not substantially affect comprehension.
- "Cannot Determine": Insufficient information to categorize.

(5) Temporal_Role: Classify the temporal focus or orientation of the citation concerning events described in the article.
Select one from:
- "Historical": Pertains primarily to past events providing context or background.
- "Current event": Pertains directly to recent or ongoing events central to the article.
- "Future projection": Provides forecasts or discusses implications for future events.
- "Retrospective analysis": Offers analysis or reflections looking back at events.
- "Other": Does not fit predefined categories.
- "Cannot Determine": Insufficient information to categorize.

(6) Citation_Type: Identify the nature and format of content presented.
Select one from:
- "Main Actor": A source directly involved in the main events of the article.
- "Statistical data": Contains numerical data, charts, or quantitative analyses.
- "Expert opinion": Presents insights, interpretations, or judgments from experts or analysts.
- "Policy document": Official government or institutional policy texts or summaries.
- "News article": Reports factual developments or events.
- "Encyclopedic or Historical analysis": In-depth exploration or scholarly assessment of historical contexts.
- "Other": Does not fit predefined categories.
- "Cannot Determine": Insufficient information to categorize.

(7) Scope: Indicate the geographical or thematic breadth covered by the citation.
Select one from:
- "National": Focuses on country-wide contexts or implications.
- "Sector-specific": Relates to particular industries or sectors (e.g., energy, healthcare).
- "Local/regional": Pertains specifically to smaller geographical areas within a country.
- "International/comparative": Involves comparisons across multiple countries or global contexts.
- "Other": Does not fit predefined categories.
- "Cannot Determine": Insufficient information to categorize.

(8) Source_Authority: Evaluate the institutional or credibility context of the source. You should analyze both the information provided by the source as well as the url or description of the source, if available.
Select one from:
- "Government source": Official governmental entities or agencies.
- "Academic source": Universities, research institutions, or peer-reviewed scholarly work.
- "Industry body": Professional or industry-specific associations or organizations.
- "Media organization": Established news media or journalism sources.
- "NGO/think tank": Non-governmental organizations, research groups, or policy institutes.
- "Individual": A person or group of people.
- "Other": Does not fit predefined categories.
- "Cannot Determine": Insufficient information to categorize.

Output your analysis structured as a list of Python dictionaries, one per citation.

Now it's your turn. Here is a summary of the article:

<article_title>
{article_title}
</article_title>

<article_summary>
{article_summary}
</article_summary>

Examine the narrative role of each of the following citations:

<citations_to_examine>
{citations_to_examine}
</citations_to_examine>

For each source, answer using the attributes defined above. Provide your output strictly in the requested format. Don't say anything else.
"""


from pydantic import BaseModel, create_model
from typing import Annotated
from annotated_types import Len

from pydantic import BaseModel, Field
from typing import List, Literal

class CitationAnalysis(BaseModel):
    ID: str
    Narrative_Function: Literal[
        "Historical Background", "Explanation", "Counterpoint", "Supporting Evidence", 
        "Demonstration of Impact", "Future Projection", "Expert Commentary", 
        "Human Perspective", "Additional Context", "Other", "Cannot Determine"
    ]
    Perspective: List[Literal[
        "Authoritative", "Informative", "Supportive", "Skeptical",
        "Against", "Neutral", "Cannot Determine"
    ]]
    Centrality: Literal["High", "Medium", "Low", "Cannot Determine"]
    Temporal_Role: Literal[
        "Historical", "Current event", "Future projection",
        "Retrospective analysis", "Other", "Cannot Determine"
    ]
    Citation_Type: Literal[
        "Statistical data", "Expert opinion", "Policy document",
        "News article", "Historical analysis", "Other", "Cannot Determine"
    ]
    Scope: Literal[
        "National", "Sector-specific", "Local/regional",
        "International/comparative", "Other", "Cannot Determine"
    ]
    Source_Authority: Literal[
        "Government source", "Academic source", "Industry body",
        "Media organization", "NGO/think tank", "Other", "Cannot Determine"
    ]


def make_citation_analyses(k: int):
    return create_model(
        'CitationAnalysesOutput',
        citations=(Annotated[list[CitationAnalysis], Len(min_length=k, max_length=k)], ...),
    )    


def reconcile_citations(citation_list_one, citation_list_two):
    one_key = 'citation_id' if 'citation_id' in citation_list_one[0] else 'citation_number'
    two_key = 'citation_id' if 'citation_id' in citation_list_two[0] else 'citation_number'
    citation_ids_one = set(map(lambda x: str(x[one_key]), citation_list_one))
    citation_ids_two = set(map(lambda x: str(x[two_key]), citation_list_two))
    common_ids = citation_ids_one.intersection(citation_ids_two)
    citation_list_one_reconciled = list(filter(lambda x: str(x[one_key]) in common_ids, citation_list_one))
    citation_list_two_reconciled = list(filter(lambda x: str(x[two_key]) in common_ids, citation_list_two)) 
    return citation_list_one_reconciled, citation_list_two_reconciled


def analyze_column(df, column_name):
    return (
        df
            .groupby(['source', 'article_title'])[column_name]
            .value_counts()
            .unstack()
            .pipe(lambda df: df.div(df.sum(axis=1), axis=0)).fillna(0)
            .groupby(level=0).mean().pipe(lambda df: df.div(df.sum(axis=1), axis=0)).fillna(0)
    )


def aggregate_sources_by_position(df, column_name, col_order, step, divide_over_columns):
    aggs = {}
    for source in ['deep_research', 'news', 'wikipedia']:
        agg_df = (
            df
            .loc[lambda df: df['source'] == source]
            .assign(c=1)
            .pivot_table(columns='num_sources_perc_bin', index=column_name, values='c', aggfunc='sum')
        )
        if divide_over_columns:
            agg_df = agg_df.pipe(lambda df: df.divide(df.sum(axis=0), axis=1))
        else:
            agg_df = agg_df.pipe(lambda df: df.divide(df.sum(axis=1), axis=0))
            
        agg_df = (
            agg_df.sort_values(step, ascending=False)
            .fillna(0)
            .loc[col_order]
        )
        aggs[source] = agg_df
    return aggs    


import inspect
function_object = globals()['reconcile_citations']
source_code = inspect.getsource(function_object)



import pandas as pd 
import numpy as np 
# load and process data
all_articles_parsed = pd.read_csv('../data/all-articles-parsed.csv')

news_df = pd.read_json('../data/all-news-articles-with-parsed-sources.json', lines=True)
news_df = news_df.rename(columns={'sources': 'citation_summaries'})

deep_research_df = pd.read_json('../data/deep-research-citation-summaries.jsonl', lines=True)
deep_research_content = pd.read_json('../data/all-deep-research-content.json', lines=True)
deep_research_df = (deep_research_df
 .merge(deep_research_content, left_on='issue', right_on='title')
 .drop(columns=['issue'])
)
deep_research_df['citation_summaries'] = (
    deep_research_df
        .assign(citation_chunks=lambda df: df.apply(lambda x: reconcile_citations(x['citation_summaries'], x['citations']), axis=1))
        .assign(citation_summaries=lambda df: df['citation_chunks'].str.get(0))
        .assign(citations=lambda df: df['citation_chunks'].str.get(1))
        [[
            'title', 
            'citation_summaries', 
            'citations'
        ]]
        .explode(['citation_summaries', 'citations'])
        .pipe(lambda df: pd.concat([df['title'].reset_index(drop=True), pd.DataFrame(df['citation_summaries'].tolist()), pd.DataFrame(df['citations'].tolist())], axis=1))
        .loc[lambda df: df['citation_id'].astype(str) == df['citation_number'].astype(str)]
        .groupby(['title'])
        .agg(list)
        .apply(lambda x: list(map(lambda y: {
            'citation_id': y[0], 
            'citation': y[1], 
            'Information': y[2], 
            'date_accessed': y[3]
        }, zip(x['citation_id'], x['text'], x['Information'], x['retrieval_date']))), axis=1)
).to_list()

wiki_df = pd.read_json('../data/all-wikipedia-citations-parsed-summarized.json', lines=True)
wiki_articles = all_articles_parsed.loc[lambda df: df['url'].str.contains('wikipedia')]#.drop(columns=['citation_summaries'])
wiki_df = wiki_df.merge(wiki_articles, on='issue')

wiki_citations = pd.read_json('../data/all-wikipedia-citations-parsed.json', lines=True)
wiki_citation_df = (
    pd.DataFrame(wiki_citations.iloc[0].to_list())
        .assign(join_key=lambda df: df['citation_file'].str.split('/').str.get(2))
)
wiki_df = (
    wiki_df
        .merge(wiki_citation_df[['citations', 'join_key']], left_on='issue', right_on='join_key')
        .drop(columns=['join_key'])
)
wiki_df['citation_summaries'] = (
    wiki_df
        .assign(citation_chunks=lambda df: df.apply(lambda x: reconcile_citations(x['citation_summaries'], x['citations']), axis=1))
        .assign(citation_summaries=lambda df: df['citation_chunks'].str.get(0))
        .assign(citations=lambda df: df['citation_chunks'].str.get(1))
        [['issue', 'citation_summaries', 'citations']]
        .explode(['citation_summaries', 'citations'])
        .pipe(lambda df: pd.concat([df['issue'].reset_index(drop=True), pd.DataFrame(df['citation_summaries'].tolist()), pd.DataFrame(df['citations'].tolist())], axis=1))
        .loc[lambda df: df['citation_id'].astype(str) == df['citation_number'].astype(str)]
        .groupby(['issue'])
        .agg(list)
        .apply(lambda x: list(map(lambda y: {
            'citation_id': y[0], 
            'citation': y[1], 
            'Information': y[2], 
            'date_accessed': y[3]
        }, zip(x['citation_id'], x['text'], x['Information'], x['retrieval_date']))), axis=1)
).to_list()
wiki_df.drop(columns=['citations'], inplace=True)


# Annotate citations
full_combined_df = pd.concat([
    deep_research_df[['title',  'content', 'citation_summaries']].assign(source='deep_research'),
    wiki_df[['title',  'content', 'citation_summaries']].assign(source='wikipedia'),
    news_df.loc[lambda df: ~df['url'].str.contains('wikipedia')][['title',  'content', 'citation_summaries']].assign(source="news")    
])
citation_batch_size = 10
if RUN_SOURCE_LABELING:
    all_annotated_citations = []
    wiki_df_to_process = full_combined_df.loc[lambda df: df['source'] == 'wikipedia']
    for _, row in tqdm(full_combined_df.iterrows(), total=len(full_combined_df)):
    # all_annotated_citations = list(filter(lambda x: x['source'] != 'wikipedia', all_annotated_citations))
    # for _, row in tqdm(wiki_df_to_process.iterrows(), total=len(wiki_df_to_process)):
        (title, content, citations, source) = row[['title', 'content', 'citation_summaries', 'source']]
        summary = prompt_openai(SUMMARY_PROMPT.format(article_title=title, article=content))
        for i in tqdm(
            range(0, len(citations), citation_batch_size), 
            total=len(citations)//citation_batch_size, 
            desc=f"Source: {source}. Processing {title}..."
        ):
            citations_batch = citations[i:i+citation_batch_size]
            citations_formatted = list(map(format_citation, citations_batch))
            prompt = ANALYSIS_PROMPT.format(
                k=len(citations_batch),
                article_title=title, 
                article_summary=summary, 
                citations_to_examine='\n\n'.join(citations_formatted)
            )
            r = prompt_openai(prompt, response_format=make_citation_analyses(len(citations_formatted)))
            annotation_list = json.loads(r.model_dump_json(indent=2))['citations']
            for citation in annotation_list:
                citation['source'] = source
                citation['article_title'] = title
                citation['article_summary'] = summary
                all_annotated_citations.append(citation)
    all_annotated_citations_df = pd.DataFrame(all_annotated_citations)
    all_annotated_citations_df.to_json('../data/FULL-annotated-citations.json', orient='records', lines=True)
else:
    all_annotated_citations_df = pd.read_json('../data/FULL-annotated-citations.json', lines=True)


import numpy as np 
step = .1
all_annotated_citations_df = (
    all_annotated_citations_df
    .assign(c=1)
    .assign(num_sources_per_story=lambda df: df.groupby(['source', 'article_title']).transform('count')['c'])
    .assign(source_idx=lambda df: df.groupby(['source', 'article_title']).cumcount() + 1)
    .drop(columns=['c'])
    .assign(num_sources_perc=lambda df: df['source_idx'] / df['num_sources_per_story'])
    .assign(num_sources_perc_bin=lambda df: pd.cut(df['num_sources_perc'], np.arange(0, 1 + step, step)).apply(lambda x: x.right).astype(float).fillna(0))
)    


all_annotated_citations_df = (full_combined_df[['title', 'source', 'citation_summaries']]
 .explode('citation_summaries')
 .reset_index(drop=True) 
 .pipe(lambda df: pd.concat([df[['source', 'title']].reset_index(drop=True), pd.DataFrame(df['citation_summaries'].tolist())], axis=1))
 .assign(ID=lambda df: df.apply(lambda x: x['citation_id'] if pd.notnull(x['citation_id']) else x['Name'], axis=1))
 .drop(columns=['citation_id', 'Name'])
 .merge(all_annotated_citations_df, left_on=['ID', 'title', 'source'], right_on=['ID','article_title','source'], how='right')
)





## unused analyses
# Scope of the first source in each story
(all_annotated_citations_df
 .groupby(['source', 'article_title']).apply(lambda df: df.iloc[0]).reset_index(drop=True)
 .pipe(analyze_column, 'Scope')
 .drop(columns=['Cannot Determine'])
 .round(3)
 .style.format(precision=3, na_rep='').background_gradient(cmap='Spectral', low=1, high=0, axis=None)
 )


# Narrative function of the first source in the story
(
    h.all_annotated_citations_df
        .groupby(['source', 'article_title']).apply(lambda df: df.iloc[0]).reset_index(drop=True)
        .explode('Narrative_Function')
        .pipe(h.analyze_column, 'Narrative_Function')
        .style.format(precision=3, na_rep='').background_gradient(cmap='Spectral', low=1, high=0, axis=None)
 )

