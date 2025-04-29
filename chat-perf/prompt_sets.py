from llmconfig.llmexchange import LLMMessagePair

galaxies_prompt = 'How many galaxies are there?'
explain_prompt = 'Explain antibiotics'
onesentence_prompt = ('Antibiotics are a type of medication used to treat bacterial infections. They work by either killing '
                      'the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. '
                      'Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes '
                      'administered intravenously. They are not effective against viral infections, and using them '
                      'inappropriately can lead to antibiotic resistance. Explain the above in one sentence:')
info_prompt = ('Author-contribution statements and acknowledgements in research papers should state clearly and specifically '
               'whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their '
               'manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers '
               'to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. '
               'Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting '
               'submitted manuscripts.  Mention the large language model based product mentioned in the paragraph above:')
teplizumab_prompt = ('Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, '
                     'scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, '
                     'the molecule was able to bind to the surface of T cells and limit their cell-killing potential. '
                     'In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the '
                     'first therapeutic antibody allowed for human use.  \nQuestion: What was OKT3 originally sourced from?')
neutralfood_prompt = 'I think the food was okay.'
blackholes_prompt = 'Can you tell me about the creation of blackholes?'
rag_lc_rlm_prompt = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved context"
                     " to answer the question. If you don't know the answer, just say that you don't know. Use three "
                     "sentences maximum and keep the answer concise.  \nQuestion: {question}  \nContext: {context}  \nAnswer:")

# each value is a list of message-sets (i.e. lists of LLMMessagePair's] to run
llm_prompt_sets = {
    'space': [
        [LLMMessagePair('user', galaxies_prompt)],
        [LLMMessagePair('user', blackholes_prompt)]
    ],
    'explain': [
        [LLMMessagePair('user', explain_prompt)]
    ],
    'onesentence': [
        [LLMMessagePair('user', onesentence_prompt)]
    ],
    'info': [
        [LLMMessagePair('user', info_prompt)]
    ],
    'drug': [
        [LLMMessagePair('user', teplizumab_prompt)]
    ],
    'gorbash-test': [
        # 'what data security does gorbash have?'
        [LLMMessagePair('user', 'gorbash compliance hotline number?')]
    ],
}
