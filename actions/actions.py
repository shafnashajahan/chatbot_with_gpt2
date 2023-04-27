# This files contains your custom actions which can be used to run

import os
import random

import pandas as pd

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#from actions.hugging_face_module.gpt2_model import *
from actions.hugging_face_module.gpt2_model import Model_Helper_Methods

# Expected to be the directory containing the action.py file
model_helpers = Model_Helper_Methods()
session_dict = {'question': " ",
                'question_category':" ",
                'advice_category': " ",
                'card_name': " "
               }
# Internal variables
fpath = "C:\\Users\\user\\Desktop\\demo_backend\\final_dataset.xlsx"
tab_lst = ['relationship']#, 'career', 'money', 'generalFuture', 'family', 'Templates']

def load_curated_data():
    intent_collection_dict = {}
    temp_lst = []
    assert os.path.exists(fpath)

    for sname in tab_lst:
        df = pd.read_excel(fpath, sheet_name=sname, index_col=0, na_filter=False)
        intent_collection_dict[sname] = df

        temp_lst.extend(list(df.index))

    TAROT_CARD_LST = list(set(temp_lst))
    # print("Final tarot keys: {}\n{}".format(TAROT_CARD_LST, '-'*70))
    return intent_collection_dict, TAROT_CARD_LST

#INTENT_TAROT_ADVICE_DATA, TAROT_CARD_LST = load_curated_data()
INTENT_TAROT__DATA, TAROT_CARD_LST = load_curated_data()

#ask user to type the questions    
class ActionPickUp(Action):
    def name(self) -> Text:
        return "action_ask_question"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('\n\n In action: {}\n\n'.format(self.name())) 
        #dispatcher.utter_message(text="Type in your question in the below space. Please enter one question at a time.") 
        dispatcher.utter_message(text="What would you like to ask. Please ask one question at a time.")
        return []


#shows list of card to the users. user can select the card from here
class ActionShowTarot(Action):
    def name(self) -> Text:
        return "action_show_tarot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print('\n\n In action: {}\n\n'.format(self.name()))
        tarot_cards_txt = ',\n'.join(TAROT_CARD_LST)
        #stores latest intent and text in a dictionary name session_dict
        session_dict['question'] = tracker.latest_message['text']
        print(session_dict['question'])
        session_dict['question_category'] = tracker.latest_message['intent']['name']
        print(session_dict['question_category'])
        session_dict['advice_category'] = session_dict['question_category'].split('_')[0]
        print(session_dict['advice_category'])
        dispatcher.utter_message(template = "utter_ask_inform_tarot")
        #dispatcher.utter_message(text="Tarot Card List:\n{}\n{}\n{}\n Pick a card from the list!\n{}".format('_'*70, tarot_cards_txt, '_'*70, '_'*70))
        return []
        
#shows selected card to the users
class ActionPickUp(Action):
    def name(self) -> Text:
        return "action_pickup_tarot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # print('\n\n In action: {}\n\n{}\n\n'.format(self.name(), tracker.latest_message))
        print('\n\n In action: {}\n\n'.format(self.name()))
 # TODO: There has to be a pattern matching logic or procedure that enables an entire phrase to be considered as the correct entit 
        session_dict['card_name'] = tracker.latest_message['text']
        #print(session_dict['card_name'] )
        #card_name = tracker.latest_message['text']
        if session_dict['card_name'] in INTENT_TAROT__DATA:
            dispatcher.utter_message(text="Invalid Tarot card: '{}'. Couldn't find in shared list.".format(session_dict['card_name']))   
        else:
            dispatcher.utter_message(text="{}\nFinally selected Tarot Card: \t{}\n{}".format(str.capitalize(session_dict['card_name'])))  
            return []
        # We could add multiple items here.
        dispatcher.utter_message(text="Are you sure, you have correctly spelled the tarot card you want?")
        return []

    
###### gpt2 model#######################################################################
# Beam-search text generation
class GPTgeneration(Action):

    def name(self) -> Text:
        return "action_text_generation"
    
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

            print('\n\n In action: {}\n\n'.format(self.name()))  

            question_category = session_dict['question_category'] #relationship_advice
            card_name = session_dict['card_name'] #the fool
            print(card_name)
            advice_category = session_dict['advice_category'] #reltaionship
            
            # assert advice_category in INTENT_TAROT__DATA
            if advice_category not in INTENT_TAROT__DATA:
                print("{}\nERROR!!\nTake a look at advice_category: {} not inINTENT_TAROT__DATA\n{}".format(advice_category))
            required_tab_df = INTENT_TAROT__DATA[advice_category]
            #print(required_tab_df)
            #print(required_tab_df.index)
            
            #assert tarot_val in required_tab_df.index
            if card_name.lower() not in required_tab_df.index:
                print("{}\nERROR!!\nTake a look at tarot_val: {} not in required_tab_df.index\n{}".format(str.capitalize(card_name), '+'*70))
                
            #fetch card name and phrases from df 
            row = required_tab_df.loc[card_name]
            phrase_text = row['light_phrase'] + ' ' + row['shadow_phrase'] + ' ' + row['relationship_advise']
            sample_question = session_dict['question'] 
            
            #call parameters of function evaluate _model from gpt2_model
            pred_ans = model_helpers.evaluate_model(sample_question, phrase_text)
            dispatcher.utter_message(text="{}\nSelected Tarot Card:\t{}\nPredicted answer:\t{}\n{}".format(session_dict['card_name'], pred_ans))
            
            #adding buttons 
            buttons = []
            buttons = [
                {"title": "Yes","payload": "/affirm", type: "postBack"}, 
                {"title": "No","payload":"/deny", type: "postBack"}
            ]
           # buttons.append({"title": 'No',"payload": 'No'})
            dispatcher.utter_message(text= "I hope this has given you some clarity about the current situation. Do you have further questions on the same topic? ", buttons=buttons)
#storing responcesin to excel
#             df1 = pd.read_excel('C:\\Users\\user\\Desktop\\demo_backend\\tarot_responses.xlsx')
#             new_row = pd.DataFrame ({'CARDS':[session_dict['card_name']],'QUESTIONS':[sample_question], 'PHRASES' : [phrase_text],'ANSWERS':[pred_ans]}, index=[0])
#             df2 = pd.concat([new_row,df1.loc[:]]).reset_index(drop=True)
#             df2.to_excel("C:\\Users\\user\\Desktop\\demo_backend\\tarot_responses.xlsx", index=False)
            return []

        
        
        
