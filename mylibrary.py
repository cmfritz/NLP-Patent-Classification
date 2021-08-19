# Module to preprocess my data

def download_file(url):
    import gc
    data = pd.read_csv(url, compression='zip', sep='\t')
    df = pd.DataFrame(data)
    del data
    gc.collect()
    return df
