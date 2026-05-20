

import logging 
from services .config import require_env ,get_env 

SEARCH_INDEX ="documents-index-v2"
MIN_SCORE =0.45 
_TOP_K =10 

_search_client =None 
_index_client =None 

def _get_search_client ():
    global _search_client 
    if _search_client is None :
        from azure .search .documents import SearchClient 
        from azure .core .credentials import AzureKeyCredential 
        _search_client =SearchClient (
        endpoint =require_env ("AZURE_SEARCH_ENDPOINT").rstrip ("/"),
        index_name =SEARCH_INDEX ,
        credential =AzureKeyCredential (require_env ("AZURE_SEARCH_KEY")),
        )
    return _search_client 

def _get_index_client ():
    global _index_client 
    if _index_client is None :
        from azure .search .documents .indexes import SearchIndexClient 
        from azure .core .credentials import AzureKeyCredential 
        _index_client =SearchIndexClient (
        endpoint =require_env ("AZURE_SEARCH_ENDPOINT").rstrip ("/"),
        credential =AzureKeyCredential (require_env ("AZURE_SEARCH_KEY")),
        )
    return _index_client 

def ensure_index ()->None :
    """Create the hybrid index with vector field if it doesn't exist."""
    from azure .search .documents .indexes .models import (
    SearchIndex ,SearchField ,SearchFieldDataType ,
    SimpleField ,SearchableField ,
    VectorSearch ,HnswAlgorithmConfiguration ,VectorSearchProfile ,
    SemanticConfiguration ,SemanticSearch ,SemanticPrioritizedFields ,
    SemanticField ,
    )
    client =_get_index_client ()
    existing =[i .name for i in client .list_indexes ()]
    if SEARCH_INDEX in existing :
        logging .info ("Index '%s' already exists.",SEARCH_INDEX )
        return 

    index =SearchIndex (
    name =SEARCH_INDEX ,
    fields =[
    SimpleField (name ="id",type =SearchFieldDataType .String ,key =True ,filterable =True ),
    SimpleField (name ="doc_id",type =SearchFieldDataType .String ,filterable =True ),
    SearchableField (name ="filename",type =SearchFieldDataType .String ,filterable =True ),
    SimpleField (name ="chunk_index",type =SearchFieldDataType .Int32 ,filterable =True ),
    SearchableField (name ="content",type =SearchFieldDataType .String ),
    SearchableField (name ="summary",type =SearchFieldDataType .String ),
    SearchField (
    name ="embedding",
    type =SearchFieldDataType .Collection (SearchFieldDataType .Single ),
    searchable =True ,
    vector_search_dimensions =1536 ,
    vector_search_profile_name ="hnsw-profile",
    ),
    SimpleField (name ="blob_url",type =SearchFieldDataType .String ),
    SimpleField (name ="uploaded_by",type =SearchFieldDataType .String ,filterable =True ),
    ],
    vector_search =VectorSearch (
    algorithms =[HnswAlgorithmConfiguration (name ="hnsw-algo")],
    profiles =[VectorSearchProfile (name ="hnsw-profile",algorithm_configuration_name ="hnsw-algo")],
    ),
    semantic_search =SemanticSearch (
    configurations =[
    SemanticConfiguration (
    name ="semantic-config",
    prioritized_fields =SemanticPrioritizedFields (
    content_fields =[SemanticField (field_name ="content")],
    keywords_fields =[SemanticField (field_name ="summary")],
    ),
    )
    ]
    ),
    )
    client .create_index (index )
    logging .info ("Index '%s' created with vector + semantic fields.",SEARCH_INDEX )

def delete_index ()->bool :
    try :
        _get_index_client ().delete_index (SEARCH_INDEX )
        logging .info ("Index '%s' deleted.",SEARCH_INDEX )
        return True 
    except Exception :
        logging .exception ("delete_index failed.")
        return False 

def index_document (
doc_id :str ,
filename :str ,
content :str ,
summary :str ,
tags :list [str ],
blob_url :str ,
embedding :list [float ]=None ,
chunk_index :int =0 ,
chunk_id :str ="",
uploaded_by :str ="",
retries :int =3 ,
)->None :
    record_id =chunk_id or doc_id 
    doc ={
    "id":record_id ,
    "doc_id":doc_id ,
    "filename":filename ,
    "chunk_index":chunk_index ,
    "content":content [:32000 ],
    "summary":summary ,
    "blob_url":blob_url ,
    "uploaded_by":uploaded_by ,
    }
    if embedding :
        doc ["embedding"]=embedding 

    from time import sleep 
    for attempt in range (1 ,retries +1 ):
        try :
            results =_get_search_client ().upload_documents (documents =[doc ])
            if results [0 ].succeeded :
                logging .info ("Indexed chunk: %s (chunk %d)",filename ,chunk_index )
                return 
            logging .warning ("index_document attempt %d/%d failed",attempt ,retries )
        except Exception as exc :
            logging .warning ("index_document attempt %d/%d: %s",attempt ,retries ,exc )
        if attempt <retries :
            sleep (2 **attempt )

    raise RuntimeError (f"Indexing failed for {record_id} after {retries} attempts")

def get_indexed_doc_ids ()->set :
    """Return the set of doc_ids that have at least one chunk in the search index."""
    try :
        results =list (_get_search_client ().search (
        search_text ="*",
        select =["doc_id"],
        top =1000 ,
        ))
        return {r ["doc_id"]for r in results if r .get ("doc_id")}
    except Exception :
        logging .exception ("get_indexed_doc_ids failed")
        return set ()

def backfill_uploaded_by (doc_id_to_owner :dict [str ,str ])->int :
    """
    Merge `uploaded_by` into existing search index documents.
    doc_id_to_owner: { doc_id: uploaded_by_email }
    Returns count of documents updated.
    """
    client =_get_search_client ()
    updated =0 
    for doc_id ,owner in doc_id_to_owner .items ():
        if not owner :
            continue 

        try :
            results =list (client .search (
            search_text ="*",
            filter =f"doc_id eq '{doc_id}'",
            select =["id"],
            top =1000 ,
            ))
            if not results :
                continue 
            patches =[{"id":r ["id"],"uploaded_by":owner }for r in results ]
            client .merge_or_upload_documents (documents =patches )
            updated +=len (patches )
            logging .info ("backfill_uploaded_by: patched %d chunks for doc_id=%s owner=%s",
            len (patches ),doc_id ,owner )
        except Exception as exc :
            logging .warning ("backfill_uploaded_by failed for doc_id=%s: %s",doc_id ,exc )
    return updated 

def vector_search (
query_embedding :list [float ],
query_text :str ,
top :int =_TOP_K ,
filename_filter :str ="",
uploaded_by :str ="",
doc_ids :list [str ]|None =None ,
min_score :float =MIN_SCORE ,
)->list [dict ]:
    """
    Hybrid retrieval:
      - VectorizedQuery (embedding) + full-text BM25 in one request
      - Semantic reranking via query_type="semantic"
      - Threshold filter: discard results below min_score
      - Returns top-K chunks
    """
    from azure .search .documents .models import VectorizedQuery 

    vector_query =VectorizedQuery (
    vector =query_embedding ,
    k_nearest_neighbors =top *6 ,
    fields ="embedding",
    )

    search_kwargs :dict ={
    "search_text":query_text ,
    "vector_queries":[vector_query ],
    "select":["id","doc_id","filename","chunk_index","content","summary","blob_url"],
    "top":top *6 ,
    }

    try :
        search_kwargs ["query_type"]="semantic"
        search_kwargs ["semantic_configuration_name"]="semantic-config"
        search_kwargs ["query_caption"]="extractive"
    except Exception :
        pass 

    if doc_ids and len (doc_ids )>0 :

        safe_ids =",".join (d .replace ("'","''")for d in doc_ids )
        doc_filter =f"search.in(doc_id, '{safe_ids}', ',')"
        if uploaded_by :
            safe =uploaded_by .replace ("'","''")
            search_kwargs ["filter"]=f"uploaded_by eq '{safe}' and {doc_filter}"
        else :
            search_kwargs ["filter"]=doc_filter 
    elif filename_filter and uploaded_by :
        safe =uploaded_by .replace ("'","''")
        fn =filename_filter .replace ("'","''")
        search_kwargs ["filter"]=f"uploaded_by eq '{safe}' and filename eq '{fn}'"
    elif filename_filter :
        search_kwargs ["filter"]=f"filename eq '{filename_filter}'"
    elif uploaded_by :
        safe =uploaded_by .replace ("'","''")
        search_kwargs ["filter"]=f"uploaded_by eq '{safe}'"

    def _run_search (kwargs :dict )->list :
        try :
            return list (_get_search_client ().search (**kwargs ))
        except Exception as exc :

            logging .warning ("Semantic search failed (%s), retrying without semantic.",exc )
            kw ={k :v for k ,v in kwargs .items ()
            if k not in ("query_type","semantic_configuration_name","query_caption")}
            try :
                return list (_get_search_client ().search (**kw ))
            except Exception :
                logging .exception ("Hybrid search failed.")
                return []

    results =_run_search (search_kwargs )

    if not results and uploaded_by :
        logging .warning (
        "vector_search: uploaded_by filter returned no results — retrying without uploaded_by"
        )
        if doc_ids and len (doc_ids )>0 :
            safe_ids =",".join (d .replace ("'","''")for d in doc_ids )
            fallback_kwargs ={**search_kwargs ,"filter":f"search.in(doc_id, '{safe_ids}', ',')"}
        elif filename_filter :

            fn =filename_filter .replace ("'","''")
            fallback_kwargs ={**search_kwargs ,"filter":f"filename eq '{fn}'"}
        else :
            fallback_kwargs ={k :v for k ,v in search_kwargs .items ()if k !="filter"}
        results =_run_search (fallback_kwargs )

    chunks =[]
    for r in results :

        reranker_score =getattr (r ,"@search.reranker_score",None )
        if reranker_score is None :
            reranker_score =r .get ("@search.reranker_score")

        bm25_score =float (r .get ("@search.score",0 )or 0 )

        if reranker_score is not None :

            score =float (reranker_score )/4.0 
        else :

            score =min (bm25_score ,1.0 )

        if score <min_score :
            continue 
        chunks .append ({
        "id":r ["id"],
        "doc_id":r .get ("doc_id",""),
        "filename":r .get ("filename",""),
        "chunk_index":r .get ("chunk_index",0 ),
        "blob_url":r .get ("blob_url",""),
        "summary":r .get ("summary",""),
        "text":r .get ("content",""),
        "score":round (score ,4 ),
        })

    chunks .sort (key =lambda x :x ["score"],reverse =True )
    top_chunks =chunks [:top ]

    logging .info ("vector_search: %d results → %d above threshold %.2f → returning %d",
    len (results ),len (chunks ),min_score ,len (top_chunks ))
    return top_chunks 
