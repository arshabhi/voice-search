from transformers.pipelines import AggregationStrategy
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import numpy as np
import regex as re



def get_key_location(extractor_model, text, keyphrases, title):
    print("keyphrases=", keyphrases)
#     text = text + "\n"
    num_chars = 300
    if not keyphrases:

        print("keyp blank")
        return text[:num_chars] + "..."
        
    for keyphrase in keyphrases:
        # re.IGNORECASE ignoring cases
        # compilation step to escape the word for all cases
        compiled = re.compile(re.escape(keyphrase), re.IGNORECASE)
        text = compiled.sub("<b>" + keyphrase + "</b>", text) # for html

    if not re.search(compiled, text):
        new_keys = extractor_model(title)
        print("extracting for", new_keys)
        return get_key_location(extractor_model, text, list(new_keys), "")

    start, end = re.search(compiled, text).span()
    print(start, end )
    start_pt = [i.end() for i in re.finditer("\n", text[:start])] or [0]
    end_pt = re.search("\n", text[end:]).start()

    
    if (start - start_pt[-1] + end_pt) > num_chars:
        print(">200")
        if (start - start_pt[-1]) < end_pt:
            return "..." + text[end + end_pt - num_chars:end + end_pt]
        return text[start_pt[-1]:start_pt[-1] + num_chars] + "..."

    return text[start_pt[-1]:end + end_pt]


def embed_text(embedding_model, documents):
    sentences  = [documents]
    sentence_embeddings = embedding_model.encode(sentences)
    sentence_embeddings = (sentence_embeddings.flatten())
    return sentence_embeddings



# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])
    
