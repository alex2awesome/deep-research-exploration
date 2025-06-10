import json
import re 
import random
from more_itertools import unique_everseen
from tqdm.auto import tqdm
import warnings
import spacy
import pandas as pd
from pydantic import BaseModel, create_model
from typing import Annotated, Literal
from annotated_types import Len
warnings.filterwarnings("ignore", category=DeprecationWarning)
from utils_helper import prompt_openai, full_combined_df

RUN_DISCOURSE_LABELING = False

DISCOURSE_LABELING_PROMPT = """
I will show you a portion of an article that is {n} sentences long, with each sentence labeled by it's index in the overall article.
I will then tell you a subset of {k} indices for you to label. 
For each of these indices, look at the corresponding sentence, and assign one of the following discourse roles to that sentence.
The goal is to describe the sentence's overall role in the structure of the article.

Choose **only one** label per sentence. Use the most specific and appropriate label. Here are the labels:

<discourse_labels>
{discourse_labels}
</discourse_labels>

Here is the article:

<article>
{article}
</article>

Here are the indices of the sentences you need to label:

<sentence_indices>
{sentence_indices}
</sentence_indices>

Return only the discourse label in your response.
"""

discourse_labels = [
    '* "Main Idea": States the central thesis or core claim of the article or section.',
    '* "Background": Provides relevant past events or context needed to understand the current topic.',
    '* "Explanation": Clarifies concepts, methods, or reasoning to support understanding of the main point.',
    '* "Supporting Evidence": Offers facts, data, or citations that reinforce a claim or argument.',
    '* "Example": Gives a specific case, anecdote, or scenario that illustrates a general point.',
    '* "Counterpoint": Presents an alternative, opposing, or contrasting viewpoint.',
    '* "Implications": Explains the significance, consequences, or broader meaning of a claim or finding.',
    '* "Transition": Connects sections or ideas, providing structural flow within the text.',
    '* "Human Perspective": Describes a personal or community experience that grounds abstract ideas in real-world impact.',
    '* "Descriptive Color": Uses tone, metaphor, humor, or stylistic elements that add flavor without contributing new factual content.',
    '* "Other": Does not fit any of the above categories.',
    '* "Cannot Determine": Too ambiguous or insufficient information to assign a category.',
]

class DiscourseSentenceLabel(BaseModel):
    sentence_idx: int
    discourse_type: Literal[
        "Main Idea", "Background", "Explanation",
        "Supporting Evidence", "Example", "Counterpoint",
        "Implications", "Transition", "Human Perspective",
        "Descriptive Color", "Other", "Cannot Determine"
    ]

def make_discourse_labeling_model(possible_labels: list[str], k: int):
    return create_model(
        'DiscourseLabelingResponse',
        discourse_labels=(Annotated[list[DiscourseSentenceLabel], Len(min_length=k, max_length=k)], ...)  # type: ignore
    )


# Load a blank English model
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("sentencizer")
nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "sentencizer"]);
# Define a function to process text through the pipeline
def sentencize_text(text: str):
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\\', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\:\s\d+', ' ', text)
    doc = nlp(text)
    # Return the processed document
    return list(map(lambda x: x.text, doc.sents))


tqdm.pandas()
full_combined_df['sentences'] = full_combined_df['content'].progress_apply(sentencize_text)




if RUN_DISCOURSE_LABELING:
    batch_size = 15

    all_discourse_labels = []
    WINDOW_LEN_BEFORE = 5
    WINDOW_LEN_AFTER = 2
    NUM_SENTENCES_START = 5
    NUM_SENTENCES_END = 5

    for index, row in tqdm(full_combined_df.iterrows(), total=len(full_combined_df)):
        sentences = enumerate(row['sentences'], start=1)
        sentences = list(map(lambda x: f'{x[0]}. {x[1]}', sentences))
        num_sentences = len(sentences)
        
        starting_sentences = sentences[:NUM_SENTENCES_START]
        ending_sentences = sentences[-NUM_SENTENCES_END:]
        for start in tqdm(
            range(0, num_sentences, batch_size), 
            total=num_sentences//batch_size, 
            desc=f"Source: {row['source']}. Processing {row['title']}..."
        ):
            # randomly skip some batches 
            if num_sentences > 300:
                if random.random() < (300 / num_sentences):
                    continue

            end = min(start + batch_size, num_sentences)
            # Create the article with the specified structure
            article = [
                *starting_sentences,
                '...',
                *sentences[max(0, start - WINDOW_LEN_BEFORE):start], # window of 3 sentences before batch start
                *sentences[start:end],
                *sentences[end:min(end + WINDOW_LEN_AFTER, num_sentences)], # window of 3 sentences after batch end
                '... ',
                *ending_sentences,
            ]
            article = list(unique_everseen(article))
            sentence_indices = list(range(start + 1, end + 1))
            random.shuffle(discourse_labels)
            prompt = DISCOURSE_LABELING_PROMPT.format(
                discourse_labels='\n'.join(discourse_labels),
                article='\n'.join(article),
                k=len(sentence_indices),
                sentence_indices='\n'.join(map(str, sentence_indices)),
                n=num_sentences
            )
            r = prompt_openai(prompt, response_format=make_discourse_labeling_model(discourse_labels, len(sentence_indices)))
            annotation_list = json.loads(r.model_dump_json(indent=2))['discourse_labels']
            for annotation in annotation_list:
                annotation['source'] = row['source']
                annotation['article_title'] = row['title']
                all_discourse_labels.append(annotation)

    all_discourse_labels_df = pd.DataFrame(all_discourse_labels)
    all_discourse_labels_df.to_csv('../data/smaller_set_discourse_labels.csv', index=False)                
else:
    all_discourse_labels_df = pd.read_csv('../data/superset_discourse_labels.csv')

import numpy as np 
all_discourse_labels_df = (
    all_discourse_labels_df
        .merge(
            full_combined_df.assign(num_sentences=lambda df: df['sentences'].str.len())[['source', 'title', 'num_sentences']],
            left_on=['source', 'article_title'],
            right_on=['source', 'title'],
            how='left'
        )
)

discourse_cols = [
        "Main Idea", 
        "Background", "Historical Background", 
        "Explanation", "Explanatory Detail",
        "Supporting Evidence", "Example", 
        "Counterpoint",
        "Implications", 
        # "Transition", 
        # "Human Perspective",
        "Descriptive Color", 
        # "Other", 
        # "Cannot Determine"
    ]
all_discourse_labels_df = (
    all_discourse_labels_df
        .loc[lambda df: df['discourse_type'].isin(discourse_cols)]
        .assign(num_sents_perc=lambda df: df['sentence_idx'] / df['num_sentences'])
        .assign(num_sents_perc_bin=lambda df: pd.cut(df['num_sents_perc'], np.arange(0, 1.05, .1)).apply(lambda x: x.right).astype(float).fillna(0))
)

def aggregate_discourse_labels(df):
    aggs = {}
    col_order = None
    for source in ['news', 'deep_research', 'wikipedia']:
        agg_df = (
            df
                .loc[lambda df: df['source'] == source]
                .assign(c=1)
                .pivot_table(columns='num_sents_perc_bin', index='discourse_type', values='c', aggfunc='sum')
                .pipe(lambda df: df.divide(df.sum(axis=1), axis=0))
                .sort_values(.1, ascending=False)
                .fillna(0)
        )
        if col_order is None:
            col_order = agg_df.index
        else:
            agg_df = agg_df.loc[col_order]
        aggs[source] = agg_df
    return aggs