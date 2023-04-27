import os
import torch
import pandas as pd
import random
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining

fpath = "C:\\Users\\shafna\\Desktop\\demo_backend\\final_dataset.xlsx"
df = pd.read_excel(fpath, sheet_name ='relationship', na_filter=False)
df1 = pd.read_excel("C:\\Users\\shafna\\Desktop\\demo_backend\\UserQuestions.xlsx", sheet_name='Sheet1', na_filter=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_fpath = "C:\\Users\\shafna\\Desktop\\demo_backend\\actions\\hugging_face_module\\gpt2_text_generated_model_10_10_22.bin"

# Defining model configuration and hyperparameters ---------------------

MAXLEN = 1024  # {768, 1024, 1280, 1600}

MODEL = 'gpt2'

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}
# ----------------------------------------------------------------------

print('\nLoading up the model...')
# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.add_special_tokens(SPECIAL_TOKENS)

# Instantiate GPT2 Model
# if SPECIAL_TOKENS:
config = AutoConfig.from_pretrained(MODEL,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)
# else:
#     config = AutoConfig.from_pretrained(MODEL,
#                                         pad_token_id=tokenizer.eos_token_id,
#                                         output_hidden_states=False)

model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

# if SPECIAL_TOKENS:
# Special tokens added, model needs to be resized accordingly
model.resize_token_embeddings(len(tokenizer))

# if model_fpath:
model.load_state_dict(torch.load(model_fpath, map_location=torch.device('cpu')))

if torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()
print('Loading complete.\n')


# def read_data():
#     data = dict()
#     # df = df.iloc[:144]
#     for idx, row in df.iterrows():
#         row['phrase'] = row['light_phrase'] + ' ' + row['shadow_phrase'] + ' ' + row['relationship_advise']
#         data[idx] = [row['Card'], row['phrase']]
#     print(f"Number of entries in data: {len(data) :,}")
#     return data


# fetching card and phrases fro read_data()
for i in range(len(df)):
    def read_data():
        data = dict()
        # df = df.iloc[:144]
        for idx, row in df.iterrows():
            row['phrase'] = row['light_phrase'] + ' ' + row['shadow_phrase'] + ' ' + row['relationship_advise']
            data[idx] = [row['Card'], row['phrase']]
        print(f"Number of entries in data: {len(data) :,}")
        return data
    data = read_data()
    index = df.index
    for i in index:
        idx = i
        card_name = data[idx][0]
        phrase_text = data[idx][1]
        #for i in range(len(df1)):
            # print(df1.loc[i, "Question"])
            # fetching question from df1

        # sample_question = df1.loc[i, "Question"]
        sample_question = "How does my future look with Adam?"
        prompt = SPECIAL_TOKENS['bos_token'] + sample_question + \
                 SPECIAL_TOKENS['sep_token'] + phrase_text + SPECIAL_TOKENS['sep_token']

        generated = torch.tensor(tokenizer.encode(prompt[:MAXLEN])).unsqueeze(0)
        generated = generated.to(device)

        model.eval()

        sample_outputs = model.generate(generated, do_sample=True, top_k=0,
                                        max_length=MAXLEN, num_beams=5,
                                        repetition_penalty=5.0,
                                        early_stopping=True,
                                        num_return_sequences=1)
        print(
            "{}\nSelected Tarot Card:\t{}\nQuestion:\t{}\nPhrase_text:\t{}".format('_' * 70, card_name, sample_question,
                                                                                   phrase_text, '_' * 70))
        # print("Prompt given to model:\n{}\n{}\nQuestion: \t\nphrase_text: \t\n{}\n".format(card_name, sample_question, phrase_text, '-' * 70))
        # model_output = []
        for i, sample_output in enumerate(sample_outputs):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)
            a = len(sample_question) + len(phrase_text)
            print("{}: {}\n\n".format(i + 1, text[a:]))
            dframe = pd.read_excel('C:\\Users\\shafna\\Desktop\\demo_backend\\tarot_responses.xlsx')
            new_row = pd.DataFrame(
                {'CARDS': [card_name], 'QUESTIONS': [sample_question], 'PHRASES': [phrase_text], 'ANSWERS': [text[a:]]},
                index=[0])
            df2 = pd.concat([new_row, dframe.loc[:]]).reset_index(drop=True)
            df2.to_excel("C:\\Users\\shafna\\Desktop\\demo_backend\\tarot_responses.xlsx", index=False)
    print("finished")