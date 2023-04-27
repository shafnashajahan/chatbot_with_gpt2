
import os
import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining
class Model_Helper_Methods:

    def __init__(self, model_fpath=''):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if not os.path.exists(model_fpath):
            self.model_fpath = os.path.join(os.getcwd(), 'actions', 'hugging_face_module', 'gpt2_text_generated_model_10_10_22.bin')
            # print(self.model_fpath)
            assert os.path.exists(self.model_fpath)
        
        # Defining model configuration and hyperparameters ---------------------

        self.MAXLEN          = 1024  #{768, 1024, 1280, 1600}

        self.MODEL           = 'gpt2'

        self.SPECIAL_TOKENS  = {"bos_token": "<|BOS|>",
                                "eos_token": "<|EOS|>",
                                "unk_token": "<|UNK|>",
                                "pad_token": "<|PAD|>",
                                "sep_token": "<|SEP|>"}
        # ----------------------------------------------------------------------

        print('\nLoading up the model...')
        # Instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)

        # Instantiate GPT2 Model
        # if self.SPECIAL_TOKENS:
        config = AutoConfig.from_pretrained(self.MODEL,
                                            bos_token_id = self.tokenizer.bos_token_id,
                                            eos_token_id = self.tokenizer.eos_token_id,
                                            sep_token_id = self.tokenizer.sep_token_id,
                                            pad_token_id = self.tokenizer.pad_token_id,
                                            output_hidden_states=False)
        # else:
        #     config = AutoConfig.from_pretrained(self.MODEL,
        #                                         pad_token_id=self.tokenizer.eos_token_id,
        #                                         output_hidden_states=False)

        self.model = AutoModelForPreTraining.from_pretrained(self.MODEL, config=config)

        # if self.SPECIAL_TOKENS:
        #Special tokens added, model needs to be resized accordingly
        self.model.resize_token_embeddings(len(self.tokenizer))

        # if self.model_fpath:
        self.model.load_state_dict(torch.load(self.model_fpath, map_location=torch.device('cpu')))

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
        print('Loading complete.\n')
        
    #Generating text with conditional beam search 
    def evaluate_model(self, sample_question, phrase_text):
        if sample_question is None:
            sample_question = "Should I leave my current job?"
        if phrase_text is None:
            phrase_text = "Freeing yourself from limitation. Expressing joy and youthful vigor. Being open-minded. Taking a leap of faith. Attuning yourself to your instincts. Being eager or curious. Exploring your potential. Embracing innovation and change. Being gullible and naive. Taking unnecessary risks. Failing to be serious when required. Being silly or distracted. Lacking experience. Failing to honor well-established traditions and limits. Behaving inappropriately."

        prompt = self.SPECIAL_TOKENS['bos_token'] + sample_question + \
                 self.SPECIAL_TOKENS['sep_token'] + phrase_text + self.SPECIAL_TOKENS['sep_token']
        print('\nQuestion: {}\n\nContext: {}\n'.format(sample_question, phrase_text))

        generated = torch.tensor(self.tokenizer.encode(prompt[:self.MAXLEN])).unsqueeze(0)
        generated = generated.to(self.device)

        self.model.eval()

        sample_outputs = self.model.generate(generated, do_sample=True, top_k=0,
                                             max_length=self.MAXLEN, num_beams=5,
                                             repetition_penalty=5.0,
                                             early_stopping=True,
                                             num_return_sequences=1)

        model_output = []
        for i, sample_output in enumerate(sample_outputs):
            text = self.tokenizer.decode(sample_output, skip_special_tokens=True)
            a = len(sample_question) + len(phrase_text)
            model_output.append(text[a:])

        return ' '.join(model_output)#is my crush attracted to me?