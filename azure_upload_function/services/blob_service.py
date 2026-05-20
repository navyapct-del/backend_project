

import os 
import logging 
import uuid 
import json 
from datetime import datetime ,timezone ,timedelta 
from azure .storage .blob import BlobServiceClient ,ContentSettings 
from services .config import require_env 

CONTAINER_NAME ="documents"
IMAGES_CONTAINER_NAME ="images"
METADATA_CONTAINER_NAME ="metadata"

class BlobService :
    def __init__ (self ):
        conn_str =require_env ("AZURE_STORAGE_CONNECTION_STRING")
        self ._client =BlobServiceClient .from_connection_string (conn_str )
        self ._ensure_container (CONTAINER_NAME )
        self ._ensure_container (IMAGES_CONTAINER_NAME )
        self ._ensure_container (METADATA_CONTAINER_NAME )

    def _ensure_container (self ,name :str ):
        container =self ._client .get_container_client (name )
        try :
            container .get_container_properties ()
        except Exception :
            logging .info ("Container '%s' not found — creating.",name )
            self ._client .create_container (name )

    def upload (self ,filename :str ,data :bytes ,content_type :str ,blob_name :str ="")->str :
        """Upload raw file bytes → returns blob URL.
        
        Images (jpg/jpeg/png) are stored in the 'images' container.
        All other files go to the 'documents' container.
        If blob_name is provided, it is used as-is (for temp uploads with custom prefix).
        Otherwise, a UUID prefix is prepended to filename to prevent collisions.
        """

        ext =filename .rsplit (".",1 )[-1 ].lower ()if "."in filename else ""
        is_image =ext in {"jpg","jpeg","png"}or content_type in {"image/jpeg","image/png"}
        container =IMAGES_CONTAINER_NAME if is_image else CONTAINER_NAME 

        if not blob_name :
            blob_name =f"{uuid.uuid4().hex}_{filename}"
        blob_client =self ._client .get_blob_client (container =container ,blob =blob_name )
        blob_client .upload_blob (
        data ,
        overwrite =True ,
        content_settings =ContentSettings (content_type =content_type ),
        )
        logging .info ("Uploaded file blob to '%s': %s",container ,blob_name )
        return blob_client .url 

    def upload_text (self ,doc_id :str ,text :str )->str :
        """
        Upload extracted text to the metadata container.
        Returns the blob URL. Falls back gracefully on failure.
        """
        blob_name =f"{doc_id}/text.txt"
        blob_client =self ._client .get_blob_client (container =METADATA_CONTAINER_NAME ,blob =blob_name )
        blob_client .upload_blob (
        text .encode ("utf-8"),
        overwrite =True ,
        content_settings =ContentSettings (content_type ="text/plain; charset=utf-8"),
        )
        logging .info ("Uploaded text blob: %s (%d chars)",blob_name ,len (text ))
        return blob_client .url 

    def upload_structured_data (self ,doc_id :str ,data :dict )->str :
        """
        Upload structured_data JSON to the metadata container.
        Returns the blob URL. Falls back gracefully on failure.
        """
        blob_name =f"{doc_id}/structured_data.json"
        blob_client =self ._client .get_blob_client (container =METADATA_CONTAINER_NAME ,blob =blob_name )
        blob_client .upload_blob (
        json .dumps (data ,ensure_ascii =False ).encode ("utf-8"),
        overwrite =True ,
        content_settings =ContentSettings (content_type ="application/json"),
        )
        logging .info ("Uploaded structured_data blob: %s",blob_name )
        return blob_client .url 

    def _get_blob_client_from_url (self ,url :str ):
        """
        Parse a blob URL and return a BlobClient.
        Works around missing get_blob_client_from_url in older SDK versions.
        URL format: https://<account>.blob.core.windows.net/<container>/<blob_name>
        """
        from azure .storage .blob import BlobClient 
        return BlobClient .from_blob_url (
        blob_url =url ,
        credential =self ._client .credential ,
        )

    def download_text (self ,text_url :str )->str :
        """Download text content from a metadata blob URL."""
        return self ._get_blob_client_from_url (text_url ).download_blob ().readall ().decode ("utf-8",errors ="replace")

    def download_structured_data (self ,sd_url :str )->dict :
        """Download and parse structured_data JSON from a metadata blob URL."""
        raw =self ._get_blob_client_from_url (sd_url ).download_blob ().readall ()
        return json .loads (raw )

    def generate_sas_url (self ,blob_url :str ,expiry_hours :int =1 )->str :
        """
        Generate a time-limited SAS URL for a private blob.
        The SAS URL allows the browser to download the file directly.
        """
        from azure .storage .blob import generate_blob_sas ,BlobSasPermissions 

        try :
            from urllib .parse import urlparse ,unquote 
            parsed =urlparse (blob_url )
            account =parsed .hostname .split (".")[0 ]
            path_parts =parsed .path .lstrip ("/").split ("/",1 )
            container =path_parts [0 ]
            blob_name =unquote (path_parts [1 ])if len (path_parts )>1 else ""

            conn_str =require_env ("AZURE_STORAGE_CONNECTION_STRING")
            account_key =None 
            for part in conn_str .split (";"):
                if part .startswith ("AccountKey="):
                    account_key =part [len ("AccountKey="):]
                    break 

            if not account_key :
                logging .warning ("generate_sas_url: could not extract AccountKey from connection string")
                return blob_url 

            sas_token =generate_blob_sas (
            account_name =account ,
            container_name =container ,
            blob_name =blob_name ,
            account_key =account_key ,
            permission =BlobSasPermissions (read =True ),
            expiry =datetime .now (timezone .utc )+timedelta (hours =expiry_hours ),
            )
            sas_url =f"{blob_url}?{sas_token}"
            logging .info ("Generated SAS URL for blob: %s (expires in %dh)",blob_name ,expiry_hours )
            return sas_url 

        except Exception as exc :
            logging .exception ("generate_sas_url failed for %s: %s",blob_url ,exc )
            return blob_url 
