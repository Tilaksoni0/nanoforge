import os 
import numpy as np
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError, GatedRepoError
import requests
import time
from huggingface_hub import HfApi,  hf_hub_download
import pyarrow.parquet as pq
from src.shared import get_base_dir
import argparse
from dotenv import load_dotenv
import tiktoken

import tokenizer 

""" This dataset driver basically does following things : 

1. Shows the available datset for the user , user can choose one or multiple and later it handles the mixing as given weights 
2. Let user pass the list of domains that he wants to have for training datset, if multiple domain passed then mixing can be done 
3. User can define a token budget for the llm training , if not default is 500m and the download also only happens appproximately till this budgt 
4. 10 percent more tokens then alloacted budget per domain are converted, and stored in the bin files. 
5. Files are stored in the data dir itself , parquets seperated from bin. This file does the pretokenisation the batch dispatch is done on top of this 


Indexes are build for the dataloader to see the bin paths and parquet paths and connection between them 
There is no particular blend happens here that part will be handeled by the dataloader. (basically blend_dataset is not here ) 
using hugging face library to download the datset. 


"""
load_dotenv() 
HF_TOKEN = os.getenv("HF_TOKEN")


BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR,"data")
idx_to_path = lambda idx: f"shard_{idx:05d}.parquet"
api = HfApi()
DATASETS_CHOICES =  {"fineweb-edu/sample-10BT" : 
                     {"repo_id" :"HuggingFaceFW/fineweb-edu" , 
                                                  "path_in_repo" : "sample/10BT", 
                                                  "repo_type" : "dataset"  

                                                  }
}


def show_available_choices(): 
    """Shows available domain from , have a fixed domain dataset currentnly will expand it to be editable """
    return DATASETS_CHOICES.keys()

def build_domain_file_index(data_dir = None): 
    """ indexing the parquet files bin files of resepective datset lives sepearetly in datset"""
    
    assert "parquet" in os.listdir(data_dir) and "bin" in os.listdir(data_dir)
    parquet_dir = os.path.join(data_dir, "parquet")
    bin_dir = os.path.join(data_dir, "bin")
    path_dict = {}

    for file_dir_name in os.listdir(parquet_dir): 
        
        file_path = os.path.join(parquet_dir,file_dir_name)
        file_names = sorted([  f for f in os.listdir(file_path) if f.endswith(".parquet") ])
        path_dict[file_dir_name] = {"parquet_paths":[os.path.join(file_path , f) for f in file_names]}

    
    for bin_file in os.listdir(bin_dir): 
        
        bin_file_stem = os.path.splitext(bin_file)[0]

        if bin_file_stem in os.listdir(parquet_dir): 
            path_dict[bin_file_stem]["bin_path"] = bin_file
        else : 
            path_dict[bin_file_stem] = bin_file
    return path_dict

      
def download_shard_with_retry(repo_id, filename, save_name, parquet_dir, max_attempts=5):
    """ download single shard from hugging face api"""
    for attempt in range(1, max_attempts + 1):
        try:
            path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=parquet_dir, token=HF_TOKEN)
            new_path = os.path.join(parquet_dir, save_name)
            os.rename(path, new_path)
            return new_path
        
        except RepositoryNotFoundError as e:
            raise RuntimeError(f"Repo '{repo_id}' not found check DATASETS_CHOICES entry for a typo or a renamed/removed dataset.") from e

        except EntryNotFoundError as e:
            raise RuntimeError(f"File '{filename}' not found in repo '{repo_id}' shard list may be stale, re-check list_repo_tree() output.") from e

        except GatedRepoError as e:
            raise RuntimeError(f"Access denied to '{repo_id}' check HF_TOKEN is set and you've accepted the dataset's terms on huggingface.co.") from e



        except (HfHubHTTPError, requests.exceptions.RequestException, OSError) as e:
           
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed to download {repo_id}/{filename} after {max_attempts} attempts: {e}")
                return None
    

def download_and_tokenize(domain, tokenizer_batch_size, tok_budget=None):
    """Downloads the single file  ,  stores it at designated path , tokenize on line sequentially so can track tokens and stop downloading when the 
    written tokens exceeeds the budget helps with waste of download and saves memory if already know the token_budget
      """
    assert domain in DATASETS_CHOICES.keys(), "Ensure that you choose domain from DATASETS_CHOICES"
    repo_info = DATASETS_CHOICES[domain]

    parquet_dir = os.path.join(DATA_DIR, "parquet", domain)
    bin_dir = os.path.join(DATA_DIR, "bin")

    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    
    parquet_paths = [p.path for p in
                      api.list_repo_tree(
                          repo_id=repo_info["repo_id"],
                          path_in_repo=repo_info["path_in_repo"],
                          repo_type="dataset",
                          recursive=True)
                      if p.path.endswith(".parquet")]

    
    remaining = tok_budget + int(0.1*tok_budget)
    file_pointer = 0
    num_downloaded_files = 0 
    tot_tokens = 0

    bin_file_path = os.path.join(bin_dir, domain + ".bin")
    with open(bin_file_path, "ab") as f_out:
        while remaining > 0 and file_pointer < len(parquet_paths):


            download_path = download_shard_with_retry(
                repo_id=repo_info["repo_id"],
                filename=parquet_paths[file_pointer],
                save_name= idx_to_path(file_pointer),
                parquet_dir = parquet_dir
            )

            if download_path is None: 
              
                print(f"skipping file {parquet_paths[file_pointer]} for domain {domain}")
                file_pointer += 1 
                continue

            pf = pq.ParquetFile(download_path)

            for batch in pf.iter_batches(batch_size=tokenizer_batch_size):
                texts = batch.to_pandas()["text"].tolist()

                
                token_lists = tokenizer.encode(texts, add_bos=True)

                for toks in token_lists:
                    arr = np.array(toks, dtype=np.uint16)
                    arr.tofile(f_out)
                    tot_tokens += len(toks)
                    remaining -= len(toks)
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

            file_pointer += 1
            num_downloaded_files+= 1 

    return {"total_tokens_written_tobin" : tot_tokens , "num_downloaded_shards" : num_downloaded_files, "allocated_budget" : tok_budget}


def main(domains , weights, tok_budget,tokenizer_batch_size):
    assert tok_budget > 0 
    assert len(domains) == len(weights) , "Ensure, to provide weights for every domain"
    assert sum(weights) == 1 
    
    tok_budget_list= list(map(round , [tok_budget*x for x in weights]))
    out = {}
    for domain,budget in zip(domains,tok_budget_list):  
        
       out_temp =  download_and_tokenize(domain,tokenizer_batch_size , tok_budget=budget ) 
       out[domain] = out_temp

    return out 
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser("Dataset mixing driver")
    parser.add_argument("--list" , action="store_true" , help="Show available domains and exit.")
    parser.add_argument("--domains" ,nargs="+" , type = str, default= None , help="Domain names, space-separated.")
    parser.add_argument("--weights",nargs = "+", type= float , default= None ,help="Mix-Weights, space-separated." )
  
    parser.add_argument("--token-budget" , type=int , default= 500_000_000 ,help= "Total training token budget")
    parser.add_argument("--tokenizer-batch-size", type=int, default=1000,help = "Batch size of tokenizer")


    

    args = parser.parse_args()


    if args.list:
        print("Available domains:")
        for choice in show_available_choices():
            print(f"  {choice}")
        exit()

    
    assert args.domains is not None and args.weights is not None, "Both --domains and --weights are required"
    assert len(args.domains) == len(args.weights) , "Domains and weights should be same in number"
    print("downloading and tokenizing files...")
    print()
    out = main(args.domains , args.weights , tok_budget=args.token_budget, tokenizer_batch_size= args.tokenizer_batch_size)
    print(f"downloaded and tokenized successfully, info:")
    print()
    print(out)