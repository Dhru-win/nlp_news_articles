
import json
import pandas as pd

# loading json file (data obtained from New york times API)
with open('response2023_08.json', 'r') as f:
    article_data = json.load(f)
    f.close()
print(json.dumps(article_data, indent=4))


def flatten_and_frame(data=article_data):

    """ 
    The function flattens the parsed json obtained and converts the data obtained to a dataframe
    """

    # getting all the necessary information from the json data that was obtained from new york times api 
    i = 0
    j = 0
    data_rows = []

    while i < len(article_data['response']['docs']): # looping through the articles
        extracted_data = {
            "abstract": article_data["response"]["docs"][i]["abstract"],
            "web_url": article_data["response"]["docs"][i]["web_url"],
            "lead_paragraph": article_data["response"]["docs"][i]["lead_paragraph"],
            "headline": article_data["response"]["docs"][i]["headline"]["main"],
            "authors": article_data["response"]["docs"][i]["byline"]["original"],
            "word_count": article_data["response"]["docs"][i]["word_count"]
        }
        # looping through the categories within the article
        while j < len(article_data['response']['docs'][i]['keywords']): 
            extracted_data[f'category{j}'] = article_data['response']['docs'][i]['keywords'][j]['value']
            j += 1
        j = 0

        data_rows.append(extracted_data)
        i += 1

    #print(data_rows)

    len(data_rows)
    article_df = pd.DataFrame(data_rows)
    return article_df

