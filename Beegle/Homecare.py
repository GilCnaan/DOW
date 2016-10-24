import pandas as pd
pd.options.display.max_columns = 50

import re
from time import time



def read_data():
    #/////////////////////READ THE DATA//////////////////////////////////////
    wipes = pd.read_csv("wipes_reviews.csv", header=0, encoding="ISO-8859-1" )
    
    #/////////////////////READ THE DATA//////////////////////////////////////
    rated_reviews = pd.read_csv("Ratings_Only_Reviews.csv", header=0, encoding="ISO-8859-1")
    
    #/////////////////////LEFT JOIN DATA ON ID///////////////////////////////
    wipes = pd.merge(wipes, rated_reviews, how='left', on='ID')
    
    #/////////////////////READ IN MARKET TERMS////////////////////////////////
    market_terms = pd.read_csv("wipes_market_item_name_terms.csv", header=0, encoding="ISO-8859-1" )
    
    return (wipes, market_terms)
	
	
def remove_duplicates(wipes):
    # Remove the reviews for "wipes warmers" and "Dispensers" as they are not part of the analysis

    #/////////////////////REMOVE THESE ITEMS/////////////////////////////////
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Warmer")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Dispenser")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Shark Navigator")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Wipes Case")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Case Kit")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Washcloth")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("popchips")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Needles")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Lunette")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Thermos")].index)
    wipes = wipes.drop(wipes[wipes['Item Info Item'].str.contains("Carriage")].index)
    
    #/////////////////////REMOVE DUPLICATES///////////////////////////////////
    wipes = wipes.drop_duplicates(subset=('Text', 'Title'), keep='last')
    
    #/////////////////////REMOVE UNUSED COLUMNS///////////////////////////////
    #wipes.columns.values
    del wipes['Unnamed: 0']
    del wipes['Subject']
    del wipes['Thread Title']
    del wipes['Author Klout Score']
    del wipes['Site Info URL']
    del wipes['Site Info Country']
    del wipes['LinksCount']
    del wipes['Review Type']
    del wipes['Search Name']
    del wipes['Links Count']

    #/////////////////////RESET INDEX/////////////////////////////////////////
    wipes.reset_index(level=None, drop=True, inplace=True)
    
    #wipes.shape
    return (wipes)
	
	
def letters_only(text):
    #/////////////////////REMOVE NON-ALPHA NUMERIC CHARACTERS/////////////////
    try:
        x = re.sub("[^a-zA-Z0-9]",              # The pattern to search for
                   " ",                         # The pattern to replace it with
                   text['Item Info Item'] )     # The text to search
    except:
        return ('byte_code_error_ignore_this_ record')
    return (x)

def make_lower(text):
    #/////////////////////MAKE LOWER CASE/////////////////////////////////////
    try:
        x = text['Product']
        x = x.lower()
        x = x.split()                           # Split into words
    except:
        return (text['Text'])
    return( " ".join( x ))
	
	
def make_terms(market_terms):
    #/////////////////////GET THE TERMS ASSOCIATED WITH THE HOME MARKET////////
    home_terms = []
    for i in market_terms[market_terms['Market']=='Home']['Term']:
        if i == 'car':
            home_terms.append('car ')
        else:
            home_terms.append(i)
        home_terms.extend(['all purpose', ' bath room' ,'hardwood', 'multi-surface','multi-use','multipurpose'])
    
    #/////////////////////CREATE A LIST OF ADDITIONAL TERMS TO EXCLUDE/////////
    exlude_list = ['baby','denture','beauty','facial','skincare','skin care']
    return (home_terms, exlude_list)
	
	
def get_home_products(text, market_terms,exlude_list):
    # create a list of words for the product
    x = text['Product']
    x = x.split()
    
    if len(set(x).intersection(home_terms)) > 0:
        if len(set(x).intersection(exlude_list)) == 0:
            return ('KEEP')
        else:
            return ('REMOVE')
    else:
        return ('REMOVE')
		
		
def group_reviews(wipes):
    
    # Group the reviews by 'Item Info Item' and count how many reviews there are for those products
    reviews_grouped = pd.DataFrame(wipes[['ID','Item Info Item']].groupby(['Item Info Item']).agg(['count']))
    # Create the Product column based on the index (which is the unique term from Item Info Item)
    reviews_grouped['Product'] = reviews_grouped.index
    # Reorder and rename the columns
    cols = ['Product', 'ID']
    reviews_grouped = reviews_grouped[cols]
    reviews_grouped.columns = ['Product', 'Count']
    
    # Reset the index on the dataframe
    reviews_grouped.reset_index(level=None, drop=True, inplace=True)

    return (reviews_grouped, reviews_grouped[reviews_grouped['Count']>10])
	
	
def at_least_ten_reviews(text, products_to_review):
    x = text['Item Info Item']
    if x in products_to_review:
        return ('Keep')

def finalize_output(wipes):
    # Delete the colums: Product, Keep, Review
    wipes = pd.DataFrame(wipes[wipes['Review']=='Keep'])
    del wipes['Product']
    del wipes['Keep']
    del wipes['Review']

    #wipes['Text'] = wipes['Text'].str.replace('\n', '')
    wipes.reset_index(level=None, drop=True, inplace=True)
    return (wipes)
	
	
def keep_records(text, products_to_review):
    x = text['Item Info Item']
    if x in products_to_review:
        return ('Keep')
    return (' ')
    
	
def write_data(wipes, reviews_grouped, more_than_ten_reviews):
    wipes.to_csv('home_products.csv', index=False)
    reviews_grouped.to_csv('count_of_products.csv', index=False)
    more_than_ten_reviews.to_csv('products_with_more_than_ten_reviews.csv', index=False)
	
	
t0 = time()

wipes, market_terms = read_data()
wipes = remove_duplicates(wipes)
wipes['Product'] = wipes.apply(lambda text: letters_only(text), axis=1)
wipes['Product'] = wipes.apply(lambda text: make_lower(text), axis=1)
home_terms, exlude_list = make_terms(market_terms)
wipes['Keep'] = wipes.apply(lambda text: get_home_products(text, home_terms, exlude_list), axis=1)

# Select only the reviews that have been marked 'KEEP,' all else is discarded
wipes = pd.DataFrame(wipes[wipes['Keep']=='KEEP'])

reviews_grouped, more_than_ten_reviews = group_reviews(wipes)
products_to_review = list(more_than_ten_reviews['Product'])
wipes['Review'] = wipes.apply(lambda text: keep_records(text, products_to_review), axis=1)
wipes = finalize_output(wipes)
write_data(wipes, reviews_grouped, more_than_ten_reviews)
print("Done in %0.3fs." % (time() - t0))