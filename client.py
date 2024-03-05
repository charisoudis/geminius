from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional

import pandas as pd
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from utils.misc import env_get, env_set

env_set('CLOUD_ML_PROJECT_ID', 'spring-monolith-415922')

from utils.rag import get_document_metadata, get_similar_text_from_query, print_text_to_text_citation, \
    get_gemini_response, get_similar_image_from_query, print_text_to_image_citation_single, Color


class Gemini:
    GCP_INITIALIZED: bool = False
    IMAGE_DESC_PROMPTS: Dict[str, str] = dict(
        explain_default="""Explain what is going on in the image.
If it's a table, extract all elements of the table.
If it's a graph, explain the findings in the graph.
Do not include any numbers that are not mentioned in the image.
""",
    )

    def __init__(self, text_model_name: str = 'gemini-1.0-pro', image_model_name: str = 'gemini-1.0-pro-vision'):
        if not self.__class__.GCP_INITIALIZED:
            vertexai.init(project=env_get('CLOUD_ML_PROJECT_ID'), location=env_get('GCP_PROJECT_LOCATION'))
            self.__class__.GCP_INITIALIZED = True
        self.text_model = GenerativeModel(text_model_name)
        self.multimodal_model = GenerativeModel(image_model_name)

    def process_pdfs(self, pdfs_dir: Path, prompt_key: str = 'explain_default') -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Extract text and image metadata from the PDF document
        return get_document_metadata(
            self.multimodal_model,
            str(pdfs_dir),
            image_save_dir=str(pdfs_dir / 'images'),
            image_description_prompt=prompt_key,
            embedding_size=1408,
            # add_sleep_after_page = True, # Uncomment this if you are running into API quota issues
            # sleep_time_after_page = 5,
        )

    # noinspection PyMethodMayBeStatic
    def text_search(self, query: str, text_metadata_df: pd.DataFrame, top_n: int = 3,
                    verbose: bool = False) -> Tuple[Dict[int, Dict[str, Any]], str]:
        # Match user text query with "chunk_embedding" to find relevant chunks
        matching_results_text = get_similar_text_from_query(
            query,
            text_metadata_df,
            column_name="text_embedding_chunk",
            top_n=top_n,
            chunk_text=True,
        )
        if verbose:
            # Print the matched text citations
            print_text_to_text_citation(matching_results_text, print_top=False, chunk_text=True)
        # All relevant text chunk found across documents based on user query
        context = "\n".join(
            [value["chunk_text"] for key, value in matching_results_text.items()]
        )
        return matching_results_text, context

    def image_search(self, query: str, text_metadata_df: pd.DataFrame, image_metadata_df: pd.DataFrame, top_n: int = 3,
                     use_image_emb: bool = False, df_col_name: str = 'text_embedding_from_image_description',
                     verbose: bool = False) -> Union[Dict[int, Dict[str, Any]], str]:
        # Search for similar images with text query
        matching_results_image = get_similar_image_from_query(
            text_metadata_df,
            image_metadata_df,
            query=query,
            column_name=df_col_name,  # Use image description text embedding
            image_emb=use_image_emb,  # Use text embedding instead of image embedding
            top_n=top_n,
            embedding_size=1408,
        )
        if verbose:
            # Print the matched text citations
            color = Color()
            for image_index, image_data in matching_results_image.items():
                print_text_to_image_citation_single(image_index, image_data, color=color)
                image_data["image_object"]._pil_image.show()
                print('\n')
        # create context
        context = []
        for matching_image_data in matching_results_image.values():
            context.extend([
                "Image: ", matching_image_data["image_object"],
                "Caption: ", matching_image_data["image_description"]
            ])
        return matching_results_image, str(context)

    def answer(self, query: str, ctx: Optional[str], temperature: float = 0.2, max_output_tokens: int = 2048) -> str:
        # Get Gemini's answer to the given question and optionally context
        return get_gemini_response(
            self.text_model,  # we are passing Gemini 1.0 Pro
            model_input=f"""Answer the question with the given context.
Question: {query}
Context: {ctx}
Answer:
""",
            stream=True,
            generation_config=GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens),
        )

    def __call__(self, query: str, pdfs_dir: Path, rag_top_n: int = 5, temperature: float = 0.4,
                 verbose: bool = False) -> str:
        # 1. Process PDF files
        text_metadata_df, image_metadata_df = self.process_pdfs(pdfs_dir)
        # 2. Search for similar text chunks in the PDFs
        text_matches, text_context = self.text_search(query, text_metadata_df, top_n=rag_top_n, verbose=verbose)
        # 3. Search for similar images in the PDFs
        image_matches, image_context = self.image_search(query, text_metadata_df, image_metadata_df, top_n=rag_top_n)
        # 4. Create LLM prompt
        prompt = f""" Instructions: Compare the images and the text provided as Context: to answer multiple Question:
        Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.
        If unsure, respond, "Not enough context to answer".

        Context:
         - Text Context:
         {text_context}
         - Image Context:
         {image_context}

        {query}

        Answer:
        """
        return get_gemini_response(
            self.multimodal_model,
            model_input=[prompt],
            stream=True,
            generation_config=GenerationConfig(temperature=temperature, max_output_tokens=2048),
        )
