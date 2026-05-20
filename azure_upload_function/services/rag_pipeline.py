

import re 
import json 
import logging 
import hashlib 
from functools import lru_cache 
from typing import Literal 

IntentType =Literal ["structured","prose","hybrid"]

_STRUCTURED_KEYWORDS ={

"sum","total","average","avg","mean","count","max","min",
"how many","number of","add up","calculate",

"breakdown","group by","per","by department","by category",
"distribution","frequency",

"compare","comparison","versus"," vs ","difference between",

"filter","where","only","exclude",

"top","bottom","highest","lowest","ranked",

"list all","show all","show me all","all the","give me all",
"enumerate","display all","fetch all","get all",
}

_CHART_KEYWORDS ={
"chart","graph","plot","visualize","visualise","visualisation",
"bar chart","line chart","pie chart","scatter","histogram",
"trend","over time","growth","distribution",
}

_TABLE_KEYWORDS ={
"table","list","show all","enumerate","tabular","rows","columns",
"spreadsheet","grid",
}

def classify_intent (query :str ,has_structured_data :bool )->IntentType :
    """
    Classify query intent to route to the right pipeline.

    Returns:
      "structured" — use query engine (aggregations, charts from data)
      "prose"      — use RAG (factual questions, summaries, explanations)
      "hybrid"     — try query engine first, fall back to RAG
    """
    q =query .lower ()

    import re as _re 
    def _has_kw (kw :str )->bool :
        if len (kw )<=4 :
            return bool (_re .search (r'\b'+_re .escape (kw )+r'\b',q ))
        return kw in q 

    has_chart_intent =any (k in q for k in _CHART_KEYWORDS )
    has_structured_intent =any (_has_kw (k )for k in _STRUCTURED_KEYWORDS )
    has_table_intent =any (k in q for k in _TABLE_KEYWORDS )

    prose_triggers ={"summarize","summary","explain","describe","what is","who is",
    "tell me about","overview","introduction","background"}
    if any (t in q for t in prose_triggers ):
        return "prose"

    if not has_structured_data :
        return "prose"

    if has_chart_intent or (has_structured_intent and has_structured_data ):
        return "structured"

    if has_table_intent :
        return "hybrid"

    return "prose"

def generate_query_variants (query :str )->list [str ]:
    """
    Generate 3 semantically diverse query variants + 1 HyDE hypothetical answer.
    Returns [original, variant1, variant2, variant3, hyde_passage].

    HyDE (Hypothetical Document Embedding): generate a short hypothetical answer
    and embed it — this often retrieves better chunks than the raw question.
    """
    from services .openai_service import _get_client ,_deployment 

    prompt =f"""Given this user question, generate:
1. One alternative phrasing (different vocabulary, same intent)
2. A short hypothetical answer passage (1-2 sentences) that would ideally answer the question

Return ONLY valid JSON:
{{"variant": "rephrased question here", "hyde": "hypothetical answer here"}}

Question: {query}

JSON:"""

    try :
        resp =_get_client ().chat .completions .create (
        model =_deployment (),
        messages =[{"role":"user","content":prompt }],
        temperature =0.3 ,
        max_tokens =150 ,
        timeout =8 ,
        )
        raw =resp .choices [0 ].message .content .strip ()
        raw =re .sub (r"^```(?:json)?\s*","",raw )
        raw =re .sub (r"\s*```$","",raw ).strip ()
        data =json .loads (raw )
        variant =data .get ("variant","")
        hyde =data .get ("hyde","")
        result =[query ]
        if variant and variant !=query :
            result .append (variant )
        if hyde :
            result .append (hyde )
        logging .info ("generate_query_variants: %d queries total",len (result ))
        return result 
    except Exception as exc :
        logging .warning ("generate_query_variants failed (%s) — using original query only",exc )
        return [query ]

def multi_query_retrieve (
query :str ,
top_k :int =7 ,
filename_filter :str ="",
uploaded_by :str ="",
use_hyde :bool =True ,
doc_ids :list [str ]|None =None ,
)->list [dict ]:
    """
    Retrieve chunks using multiple query variants + HyDE for better recall.
    Embeddings are generated in parallel using ThreadPoolExecutor for speed.
    """
    from services .openai_service import generate_embedding 
    from services .search_service import vector_search 
    from concurrent .futures import ThreadPoolExecutor ,as_completed 

    if use_hyde :
        queries =generate_query_variants (query )
    else :
        queries =[query ]

    embeddings :dict [str ,list [float ]]={}
    with ThreadPoolExecutor (max_workers =len (queries ))as executor :
        future_to_q ={executor .submit (generate_embedding ,q ):q for q in queries }
        for future in as_completed (future_to_q ):
            q =future_to_q [future ]
            emb =future .result ()
            if emb :
                embeddings [q ]=emb 

    seen_ids :dict [str ,dict ]={}

    for q_variant ,embedding in embeddings .items ():
        chunks =vector_search (
        query_embedding =embedding ,
        query_text =q_variant ,
        top =top_k ,
        filename_filter =filename_filter ,
        uploaded_by =uploaded_by ,
        doc_ids =doc_ids ,
        )
        for chunk in chunks :
            cid =chunk ["id"]
            if cid not in seen_ids or chunk ["score"]>seen_ids [cid ]["score"]:
                seen_ids [cid ]=chunk 

    if not seen_ids :

        embedding =generate_embedding (query )
        if embedding :
            return vector_search (
            query_embedding =embedding ,
            query_text =query ,
            top =top_k ,
            filename_filter =filename_filter ,
            uploaded_by =uploaded_by ,
            doc_ids =doc_ids ,
            )
        return []

    merged =sorted (seen_ids .values (),key =lambda x :x ["score"],reverse =True )

    if doc_ids and len (doc_ids )>1 :
        per_doc :dict [str ,list ]={}
        for chunk in merged :
            did =chunk .get ("doc_id","")
            per_doc .setdefault (did ,[]).append (chunk )

        guaranteed :list [dict ]=[]
        per_doc_guarantee =max (2 ,top_k //len (doc_ids ))
        for did in doc_ids :
            guaranteed .extend (per_doc .get (did ,[])[:per_doc_guarantee ])

        guaranteed_ids ={c ["id"]for c in guaranteed }
        for chunk in merged :
            if len (guaranteed )>=top_k :
                break 
            if chunk ["id"]not in guaranteed_ids :
                guaranteed .append (chunk )
                guaranteed_ids .add (chunk ["id"])

        result =guaranteed [:top_k +len (doc_ids )]
        logging .info ("multi_query_retrieve: multi-doc guarantee → %d chunks across %d docs",
        len (result ),len (doc_ids ))
    else :
        result =merged [:top_k ]

    logging .info ("multi_query_retrieve: %d unique chunks from %d queries → returning %d",
    len (seen_ids ),len (queries ),len (result ))
    return result 

def compress_chunks (query :str ,chunks :list [dict ])->list [dict ]:
    """
    Extract only the passage most relevant to the query from each chunk.
    For multi-doc queries, instructs the extractor to preserve cross-doc
    comparative content that may not be relevant in isolation.
    Falls back to original chunk text if compression fails.
    """
    from services .openai_service import _get_client ,_deployment 

    if not chunks :
        return chunks 

    if len (chunks )<=5 :
        logging .info ("compress_chunks: skipping (chunks=%d)",len (chunks ))
        return chunks 

    unique_sources ={c .get ("filename","")for c in chunks }
    is_multi_doc =len (unique_sources )>1 

    chunk_texts =[]
    for i ,chunk in enumerate (chunks ):
        text =(chunk .get ("text")or chunk .get ("content")or "").strip ()
        filename =chunk .get ("filename",f"doc{i+1}")
        chunk_texts .append (f"[Chunk {i+1} — {filename}]\n{text[:1500]}")

    combined ="\n\n".join (chunk_texts )

    multi_doc_note =(
    "\nNOTE: Chunks come from MULTIPLE documents. For comparison or overview questions, "
    "preserve content from ALL sources even if it seems redundant in isolation."
    if is_multi_doc else ""
    )

    prompt =f"""You are a precise information extractor.

For each chunk below, extract ONLY the sentences directly relevant to the question.
If a chunk has no relevant content, return an empty string for it.
Return ONLY valid JSON: {{"extracts": ["extract1", "extract2", ...]}}
The array must have exactly {len(chunks)} elements (one per chunk, empty string if not relevant).
{multi_doc_note}

Question: {query}

Chunks:
{combined}

JSON:"""

    try :
        resp =_get_client ().chat .completions .create (
        model =_deployment (),
        messages =[{"role":"user","content":prompt }],
        temperature =0.0 ,
        max_tokens =1500 ,
        timeout =10 ,
        )
        raw =resp .choices [0 ].message .content .strip ()
        raw =re .sub (r"^```(?:json)?\s*","",raw )
        raw =re .sub (r"\s*```$","",raw ).strip ()
        data =json .loads (raw )
        extracts =data .get ("extracts",[])

        compressed =[]
        for i ,chunk in enumerate (chunks ):
            extract =extracts [i ].strip ()if i <len (extracts )else ""
            if extract :

                new_chunk =dict (chunk )
                new_chunk ["text"]=extract 
                new_chunk ["_compressed"]=True 
                new_chunk ["_original_text_len"]=len (chunk .get ("text")or chunk .get ("content")or "")
                compressed .append (new_chunk )
            else :

                compressed .append (chunk )

        relevant =[
        c for c in compressed 
        if (c .get ("text")or c .get ("content")or "").strip ()
        ]

        logging .info ("compress_chunks: %d → %d relevant chunks after compression",
        len (chunks ),len (relevant ))
        return relevant if relevant else chunks 

    except Exception as exc :
        logging .warning ("compress_chunks failed (%s) — using original chunks",exc )
        return chunks 

_SYSTEM_PROMPT ="""You are a helpful, precise AI assistant for document question-answering.

CORE RULES:
1. Answer ONLY using the provided document context. Do not use external knowledge.
2. If the context does not contain the answer, say: "No relevant information found in this document."
3. Be specific and factual. Include exact figures, names, and dates from the context.
4. Do not fabricate data, names, or statistics not present in the context.

RESPONSE QUALITY:
- Be comprehensive but concise
- Use numbered lists for multi-part answers
- Use bullet points for enumerations
- Bold key terms or figures using **term** syntax"""

_SYSTEM_PROMPT_MULTI ="""You are a helpful, precise AI assistant for multi-document question-answering.

CORE RULES:
1. Answer ONLY using the provided document context. Do not use external knowledge.
2. Each source is labelled [Source N: filename]. Always attribute facts to their source.
3. If a document does not contain relevant information, explicitly state that for that document.
4. Do not fabricate data, names, or statistics not present in the context.

RESPONSE QUALITY:
- Address each document separately before giving a combined answer when relevant
- Use numbered lists for multi-part answers
- Use bullet points for enumerations
- Bold key terms or figures using **term** syntax
- End with a brief cross-document summary if the question asks for comparison or overview"""

def grounded_generate (
query :str ,
chunks :list [dict ],
response_format :Literal ["text","table","chart","auto"]="auto",
history :list [dict ]|None =None ,
)->dict :
    """
    Generate a grounded answer from compressed, relevant chunks.

    Args:
        query:           User question
        chunks:          Compressed, relevant chunks from retrieval
        response_format: Force a specific format or let the LLM decide

    Returns:
        {
          "type":    "text" | "table" | "chart",
          "answer":  str,
          "columns": [...],   # for table
          "rows":    [...],   # for table
          "labels":  [...],   # for chart
          "values":  [...],   # for chart
          "chart_type": str,  # for chart
        }
    """
    from services .openai_service import _get_client ,_deployment 

    if not chunks :
        return {
        "type":"text",
        "answer":"No relevant information found in this document.",
        }

    context_parts =[]
    citation_list =[]
    seen_files =set ()

    for i ,chunk in enumerate (chunks ,1 ):
        filename =chunk .get ("filename",f"Document {i}")
        text =(chunk .get ("text")or chunk .get ("content")or "").strip ()
        summary =chunk .get ("summary","")
        score =chunk .get ("score",0 )

        if not text :
            continue 

        if filename not in seen_files and summary :
            context_parts .append (
            f"[Source {i}: {filename} | relevance: {score:.2f}]\n"
            f"Document summary: {summary}\n\n"
            f"Relevant excerpt:\n{text}"
            )
        else :
            context_parts .append (
            f"[Source {i}: {filename} | relevance: {score:.2f}]\n{text}"
            )

        if filename not in seen_files :
            seen_files .add (filename )
            citation_list .append (filename )

    if not context_parts :
        return {
        "type":"text",
        "answer":"No relevant information found in this document.",
        }

    context ="\n\n---\n\n".join (context_parts )

    q_lower =query .lower ()
    if response_format =="auto":
        wants_chart =any (k in q_lower for k in _CHART_KEYWORDS )
        wants_table =any (k in q_lower for k in _TABLE_KEYWORDS )
        if wants_chart :
            response_format ="chart"
        elif wants_table :
            response_format ="table"
        else :
            response_format ="text"

    format_instructions =_build_format_instructions (response_format ,citation_list )

    is_multi_doc =len (seen_files )>1 
    multi_doc_instruction =""
    if is_multi_doc :
        source_list ="\n".join (f"- {f}"for f in citation_list )
        multi_doc_instruction =f"""
IMPORTANT — Multiple documents detected:
{source_list}

Structure your answer as:
1. Answer from each document separately (labelled by filename)
2. Combined summary / comparison at the end (if relevant to the question)
"""

    user_prompt =f"""Context from documents:
{context}

---

Question: {query}
{multi_doc_instruction}
{format_instructions}"""

    try :

        system_prompt =_SYSTEM_PROMPT_MULTI if is_multi_doc else _SYSTEM_PROMPT 
        messages =[{"role":"system","content":system_prompt }]
        if history :

            for turn in history [-6 :]:
                role =turn .get ("role","user")
                content =turn .get ("content","")
                if role in ("user","assistant")and content :
                    messages .append ({"role":role ,"content":content [:800 ]})
        messages .append ({"role":"user","content":user_prompt })

        resp =_get_client ().chat .completions .create (
        model =_deployment (),
        messages =messages ,
        temperature =0.0 ,
        max_tokens =2500 ,
        frequency_penalty =0.1 ,
        timeout =25 ,
        )
        raw =resp .choices [0 ].message .content .strip ()

        parsed =_parse_llm_response (raw ,citation_list ,response_format )

        if parsed .get ("type")=="text"and parsed .get ("answer"):
            parsed ["answer"]=_fix_numbering (parsed ["answer"])

        if parsed .get ("type")=="chart":
            parsed =_clean_chart_data (parsed ,user_query =query )

        logging .info ("grounded_generate: type=%s answer_len=%d",
        parsed .get ("type"),len (str (parsed .get ("answer",""))))
        return parsed 

    except Exception as exc :
        logging .error ("grounded_generate failed: %s",exc )
        return {
        "type":"text",
        "answer":"Failed to generate answer. Please try again.",
        }

def _clean_chart_data (chart :dict ,user_query :str ="")->dict :
    """
    Post-process chart data for quality.
    Respects explicit user chart type requests — never downgrades pie→bar
    if the user explicitly asked for a pie chart.
    """
    chart_type =chart .get ("chart_type","bar")
    labels =chart .get ("labels",[])
    values =chart .get ("values",[])
    _q =user_query .lower ()
    user_wants_pie =any (k in _q for k in ("pie chart","pie graph"," pie ","as pie","a pie"))

    if not labels or not values :
        return chart 

    min_len =min (len (labels ),len (values ))
    labels =labels [:min_len ]
    values =values [:min_len ]

    clean_values =[]
    for v in values :
        try :
            clean_values .append (float (v )if v is not None else 0.0 )
        except (TypeError ,ValueError ):
            clean_values .append (0.0 )

    if chart_type =="pie":

        filtered =[(l ,v )for l ,v in zip (labels ,clean_values )if v >0 ]
        if not filtered :
            chart ["labels"]=labels 
            chart ["values"]=clean_values 
            return chart 

        if len (filtered )==1 and not user_wants_pie :
            chart ["chart_type"]="bar"
            chart ["labels"]=[f [0 ]for f in filtered ]
            chart ["values"]=[f [1 ]for f in filtered ]
            return chart 
        chart ["labels"]=[f [0 ]for f in filtered ]
        chart ["values"]=[f [1 ]for f in filtered ]
    else :
        chart ["labels"]=labels 
        chart ["values"]=clean_values 

    return chart 

def _fix_numbering (text :str )->str :
    """Re-sequence any numbered list in the answer so numbers are strictly 1,2,3,..."""
    lines =text .split ("\n")
    counter =0 
    result =[]
    for line in lines :
        m =re .match (r"^(\s*)(\d+)\.\s+(.+)",line )
        if m :
            counter +=1 
            result .append (f"{m.group(1)}{counter}. {m.group(3)}")
        else :
            if line .strip ()=="":
                counter =0 
            result .append (line )
    return "\n".join (result )

def _build_format_instructions (
response_format :str ,
citation_list :list [str ],
)->str :
    """Build format-specific instructions for the LLM."""

    if response_format =="table":
        return """Respond with ONLY a valid JSON object in this exact format:
{"type":"table","columns":["Col1","Col2","Col3"],"rows":[{"Col1":"v1","Col2":"v2","Col3":"v3"}],"answer":"1-line summary of the table"}

Rules:
- Extract ALL relevant data from the context into table rows
- Use the actual column names from the data
- Include every relevant row — do not truncate
- "answer" should be a 1-line summary of what the table shows"""

    if response_format =="chart":
        return """Respond with ONLY a valid JSON object in this exact format:
{"type":"chart","chart_type":"bar|line|pie|area|scatter","labels":["A","B","C"],"values":[10,20,30],"answer":"1-line summary"}

Rules:
- Extract numeric data from the context
- IMPORTANT: If the user explicitly asked for a specific chart type (e.g. "pie chart", "line chart", "bar chart"), use EXACTLY that chart_type
- Otherwise choose chart_type based on data: line for trends/time, pie for proportions/shares, bar for comparisons
- labels = category names or time periods
- values = corresponding numeric values
- If multiple series exist, use: {"type":"chart","chart_type":"bar","series":{"Series1":[1,2,3],"Series2":[4,5,6]},"labels":["A","B","C"],"answer":"..."}"""

    return """Respond with ONLY a valid JSON object in this exact format:
{"type":"text","answer":"<your detailed answer here>"}

Rules:
- Answer must be comprehensive and directly address the question
- For multi-part answers use strictly sequential numbered lists: 1. first point\\n2. second point\\n3. third point (never restart numbering, never skip numbers)
- Use **bold** for key terms and figures
- Include exact numbers/dates/names from the context
- Separate each numbered point with a newline character \\n
- If the answer is not found in the context, say: "No relevant information found in this document." """

def _parse_llm_response (
raw :str ,
citation_list :list [str ],
expected_format :str ,
)->dict :
    """
    Parse LLM response into a structured dict.
    Handles: valid JSON, JSON in markdown fences, plain text fallback.
    """

    cleaned =re .sub (r"^```(?:json)?\s*","",raw )
    cleaned =re .sub (r"\s*```$","",cleaned ).strip ()

    try :
        parsed =json .loads (cleaned )
        if isinstance (parsed ,dict )and "type"in parsed :

            parsed .pop ("sources",None )
            return parsed 
    except Exception :
        pass 

    m =re .search (r'\{[\s\S]*\}',cleaned )
    if m :
        try :
            parsed =json .loads (m .group ())
            if isinstance (parsed ,dict )and "type"in parsed :
                parsed .pop ("sources",None )
                return parsed 
        except Exception :
            pass 

    answer_text =raw .strip ().lstrip ("{[").rstrip ("}]").strip ()
    if not answer_text :
        answer_text =raw 
    logging .warning ("_parse_llm_response: LLM returned plain text, wrapping as text response (len=%d)",len (answer_text ))
    return {
    "type":"text",
    "answer":answer_text ,
    }

_pipeline_cache :dict [str ,tuple [dict ,float ]]={}
_MAX_CACHE_SIZE =300 
_CACHE_TTL_SECS =1800 

def run_rag_pipeline (
query :str ,
filename_filter :str ="",
uploaded_by :str ="",
session_id :str ="",
top_k :int =7 ,
use_hyde :bool =True ,
use_compression :bool =True ,
doc_ids :list [str ]|None =None ,
history :list [dict ]|None =None ,
)->dict :
    """
    Full advanced RAG pipeline entry point.

    Pipeline:
      1. Multi-query retrieval with HyDE
      2. Contextual compression
      3. Grounded generation with strict system prompt
      4. Response formatting

    Returns a response dict with type, answer, sources, and optional
    table/chart data.
    """
    from services .table_service import TableService 

    if not query or not query .strip ():
        return {"type":"text","answer":"No question provided."}

    import time as _time 

    table_svc =TableService ()

    doc_ids_key ="|".join (sorted (doc_ids ))if doc_ids else ""
    freshness_key =""
    try :
        docs =table_svc .list_documents ()
        relevant =[d for d in docs if not doc_ids or d ["id"]in (doc_ids or [])]
        if relevant :
            freshness_key =max (d .get ("created_at","")for d in relevant )
    except Exception :
        pass 
    cache_key =hashlib .md5 (
    (query +"|"+filename_filter +"|"+uploaded_by +"|"+session_id +"|"+doc_ids_key +"|"+freshness_key ).encode ()
    ).hexdigest ()

    cached =_pipeline_cache .get (cache_key )
    if cached :
        result ,ts =cached 
        if _time .time ()-ts <_CACHE_TTL_SECS :
            logging .info ("run_rag_pipeline: cache hit")
            return result 
        del _pipeline_cache [cache_key ]

    chunks =multi_query_retrieve (
    query =query ,
    top_k =top_k ,
    filename_filter =filename_filter ,
    uploaded_by =uploaded_by ,
    use_hyde =use_hyde ,
    doc_ids =doc_ids ,
    )

    if not chunks :
        return {
        "type":"text",
        "answer":"No relevant information found in this document.",
        }

    q_lower =query .lower ()
    q_words =set (w .strip (".,!?;:()")for w in q_lower .split ()if len (w )>=3 )

    def _stem (w :str )->str :
        if w .endswith ("ing")and len (w )>5 :return w [:-3 ]
        if w .endswith ("ies")and len (w )>4 :return w [:-3 ]+"y"
        if w .endswith ("es")and len (w )>4 :return w [:-2 ]
        if w .endswith ("s")and len (w )>3 :return w [:-1 ]
        return w 

    stemmed_q_words ={_stem (w )for w in q_words }|q_words 

    def _sd_relevance_score (sd :dict ,filename :str )->int :
        score =0 
        sd_columns =sd .get ("columns",[])
        if not sd_columns and sd .get ("sheets"):
            for sheet_data in sd ["sheets"].values ():
                sd_columns =sheet_data .get ("columns",[])
                if sd_columns :
                    break 
        for col in sd_columns :
            col_lower =col .lower ().replace ("_"," ").replace ("-"," ")
            col_words =set (col_lower .split ())
            stemmed_col_words ={_stem (w )for w in col_words }|col_words 
            for qw in stemmed_q_words :
                if qw in col_lower or any (qw in cw or cw in qw for cw in stemmed_col_words ):
                    score +=3 
                    break 
        fname_lower =filename .lower ().replace ("_"," ").replace ("-"," ")
        for qw in stemmed_q_words :
            if qw in fname_lower :
                score +=5 
                break 
        chart_agg_kw ={"chart","graph","plot","total","sum","average",
        "count","max","min","distribution","breakdown",
        "how many","number of","how much","avg","mean"}
        if any (k in q_lower for k in chart_agg_kw )and sd_columns :
            score +=1 

        if any (k in q_lower for k in _STRUCTURED_KEYWORDS )and sd_columns :
            score +=1 
        return score 

    seen_doc_ids :set [str ]=set ()
    candidate_docs :list [tuple [str ,str ,str ]]=[]
    for chunk in chunks :
        fname =chunk .get ("filename","")
        doc_id =chunk .get ("doc_id","")
        if not fname :
            continue 
        dedup_key =doc_id or fname 
        if dedup_key in seen_doc_ids :
            continue 
        seen_doc_ids .add (dedup_key )
        candidate_docs .append ((doc_id ,fname ))

    relevant_sds :list [tuple [str ,dict ,int ]]=[]
    for doc_id ,fname in candidate_docs :

        sd =table_svc .get_structured_data (fname ,session_id =session_id ,doc_id =doc_id )
        if not sd :
            continue 
        score =_sd_relevance_score (sd ,fname )
        logging .info ("run_rag_pipeline: structured data '%s' (doc_id=%s) score=%d",fname ,doc_id ,score )
        if score >0 :
            relevant_sds .append ((fname ,sd ,score ))

    stored_sd :dict |None =None 
    if relevant_sds :
        if len (relevant_sds )==1 :

            stored_sd =relevant_sds [0 ][1 ]
            logging .info ("run_rag_pipeline: single structured doc '%s'",relevant_sds [0 ][0 ])
        else :

            stored_sd =_merge_structured_data (relevant_sds )
            logging .info ("run_rag_pipeline: merged %d structured docs",len (relevant_sds ))

    if not relevant_sds :
        logging .info ("run_rag_pipeline: no relevant structured data — using prose RAG")

    intent =classify_intent (query ,has_structured_data =stored_sd is not None )
    logging .info ("run_rag_pipeline: intent=%s has_sd=%s",intent ,stored_sd is not None )

    if intent =="structured"and stored_sd :
        from services .query_engine import generate_plan ,execute_plan ,structured_to_df 

        def _run_engine_local (query :str ,structured :dict )->dict |None :
            try :
                df =structured_to_df (structured )
                if df .empty :
                    return None 

                drop_cols =[c for c in df .columns if c .startswith ("_")and c !="_source"]
                df =df .drop (columns =drop_cols ,errors ="ignore")
                cols =list (df .columns )
                plan =generate_plan (query ,cols )
                result =execute_plan (df ,plan )
                return result 
            except Exception as exc :
                logging .warning ("_run_engine_local failed: %s",exc )
                return None 

        engine_result =_run_engine_local (query ,stored_sd )
        if engine_result and engine_result .get ("type")!="error":
            rows =engine_result .get ("rows",[])
            if rows or engine_result .get ("type")=="text":
                if engine_result .get ("type")!="chart":
                    from services .query_engine import promote_to_chart 
                    try :
                        df =structured_to_df (stored_sd )
                        drop_cols =[c for c in df .columns if c .startswith ("_")and c !="_source"]
                        df =df .drop (columns =drop_cols ,errors ="ignore")
                        cols =list (df .columns )
                        plan =generate_plan (query ,cols )
                        if any (k in query .lower ()for k in _CHART_KEYWORDS )and plan .get ("group_by"):
                            engine_result =promote_to_chart (engine_result ,query )
                    except Exception :
                        pass 
                if engine_result .get ("type")=="chart"and engine_result .get ("labels"):
                    engine_result =_clean_chart_data (engine_result ,user_query =query )
                result ={k :v for k ,v in engine_result .items ()if k !="sources"}
                _cache_result (cache_key ,result )
                return result 
        logging .info ("run_rag_pipeline: structured engine failed — falling back to prose RAG")

    if use_compression and len (chunks )>2 :
        compressed_chunks =compress_chunks (query ,chunks )
    else :
        compressed_chunks =chunks 

    if intent =="hybrid":
        response_format ="table"
    elif any (k in q_lower for k in _CHART_KEYWORDS ):
        response_format ="chart"
    elif any (k in q_lower for k in _TABLE_KEYWORDS ):
        response_format ="table"
    else :
        response_format ="text"

    result =grounded_generate (
    query =query ,
    chunks =compressed_chunks ,
    response_format =response_format ,
    history =history ,
    )

    _cache_result (cache_key ,result )
    return result 

def _merge_structured_data (docs :list [tuple [str ,dict ,int ]])->dict :
    """
    Merge structured data from multiple documents into a single dict.
    Adds a '_source' column with the filename so the query engine and user
    can tell which row came from which document.

    docs: list of (filename, structured_data, score) — sorted by score desc.
    Returns a merged structured_data dict with 'rows' and 'columns'.
    """
    from services .query_engine import structured_to_df 
    import pandas as pd 

    docs_sorted =sorted (docs ,key =lambda x :x [2 ],reverse =True )
    frames =[]
    for fname ,sd ,_ in docs_sorted :
        try :
            df =structured_to_df (sd )
            if df .empty :
                continue 

            df =df .drop (columns =[c for c in df .columns if c .startswith ("_")],errors ="ignore")
            df ["_source"]=fname 
            frames .append (df )
        except Exception as exc :
            logging .warning ("_merge_structured_data: failed to convert '%s': %s",fname ,exc )

    if not frames :
        return {}

    merged =pd .concat (frames ,ignore_index =True ,sort =False )

    merged =merged .where (pd .notnull (merged ),None )

    return {
    "rows":merged .to_dict (orient ="records"),
    "columns":list (merged .columns ),
    "_merged":True ,
    "_sources":[f for f ,_ ,_ in docs_sorted ],
    }

def _cache_result (key :str ,result :dict )->None :
    """Cache result with TTL and LRU eviction."""
    import time as _time 
    global _pipeline_cache 
    _pipeline_cache [key ]=(result ,_time .time ())
    if len (_pipeline_cache )>_MAX_CACHE_SIZE :

        keys_to_evict =list (_pipeline_cache .keys ())[:30 ]
        for k in keys_to_evict :
            _pipeline_cache .pop (k ,None )
