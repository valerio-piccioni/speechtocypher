import speech_recognition as sr
import pyaudio
import wave
from flash.audio import SpeechRecognition
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from tabulate import tabulate
from flash.core.data.utils import download_data
import wordtodigits
import speech_recognition as sr
import pyaudio
import ply.yacc as yacc

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

def movie_graph_from_cypher(data):
    G = nx.DiGraph()
    def add_node(node, key):
        # Adds node id it hasn't already been added
        if key == 'movie':
            u = node['title']
        else:
            u = node['name']
        if G.has_node(u):
            return
        G.add_node(u, labels=[key], properties=node)

    def add_edge(relation):
        # Adds edge if it hasn't already been added.
        # Make sure the nodes at both ends are created
        add_node(relation[0],'person')
        add_node(relation[2],'movie')
        # Check if edge already exists
        u = relation[0]['name']
        v = relation[2]['title']
        if G.has_edge(u, v):
            return
        # If not, create it
        G.add_edge(u, v, type_=relation[1])

    for d in data:
        for key,entry in d.items():
            # Parse node
            if key != 'r':
                add_node(entry,key)
            else:
                add_edge(entry)
    return G

class DriverNeo4j:

    def __init__(self, uri, database, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def cypherQuery(self, query):
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.data()

    def cypherQueryToDataFrame(self, query):
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return pd.DataFrame(result.data(), columns=result.keys())

    def cypherQueryToDataFrameAndPrint(self, query):
        return print(pdtabulate(self.cypherQueryToDataFrame(query)))

pdtabulate=lambda df:tabulate(df,headers='keys')
driver = DriverNeo4j("bolt://localhost:7687", "neo4j", "neo4j", "admin")

# Tokens
reserved = {
'node':'NODE',
'label':'LABEL',
'map':'MAP',
'key':'KEY',
'value':'VALUE',
'relation':'RELATION',
'right':'RIGHT',
'left':'LEFT',
'type':'TYPE',
'length':'LENGTH',
'start':'START',
'end':'END',
'id':'ID',
'match':'MATCH',
'return':'RETURN',
'dot':'DOT',
'where':'WHERE',
'not':'NOT',
'or':'OR',
'and':'AND',
'xor':'XOR',
'equals':'EQUALS',
'less':'LESS',
'greater':'GREATER',
'true':'TRUE',
'false':'FALSE',
'plus':'PLUS',
'minus':'MINUS',
'multiply':'MULTIPLY',
'divide':'DIVIDE',
'ends':'ENDS',
'starts':'STARTS',
'with':'WITH',
'contains':'CONTAINS',
'create':'CREATE',
'delete':'DELETE',
'detach':'DETACH',
'remove':'REMOVE',
'merge':'MERGE',
'set':'SET'
}

tokens = ['NUMBER'] + list(reserved.values())

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_KEYWORD(t):
    r'[a-zA-Z0-9_][a-zA-Z0-9_]*|\*'
    t.type = reserved.get(str(t.value).lower(),'ID')    # Check for reserved words
    return t

#def t_DOT_IDENTIFIER(t):
#    r'[[a-zA-Z0-9_][a-zA-Z0-9_]*(\.[a-z0-9]+)*'
#    t.type = reserved.get(str(t.value).lower(),'DOT_ID')    # Check for reserved words
#    return t
# Ignored characters
t_ignore = " \t"

def t_error(t):
    print(f"Illegal character {t.value[0]!r}")
    t.lexer.skip(1)

# Build the lexer
import ply.lex as lex
lex.lex()

# GRAMMAR START
# Precedence rules for the arithmetic operators
precedence = (
    ('left','ID','DOT'),
    )

names = {'true':'TRUE', 'false':'FALSE'}

def  p_query_expr(p):
    '''query : clause_statement_list'''
    p[0] = p[1]
    #print(p[0])
    names = {'true':'TRUE', 'false':'FALSE'}
    return p[0]

def p_clause_statement_list_expr(p):
    '''clause_statement_list : clause_statement_list clause_statement_value
                             | clause_statement_value'''

    if len(p) == 3:
        p[0] = str(p[1])+' '+str(p[2])
    else:
        p[0] = p[1]

def  p_clause_statement_value_expr(p):
    '''clause_statement_value : MATCH statement
                              | WHERE condition
                              | CREATE statement
                              | MERGE statement
                              | SET condition
                              | DETACH DELETE comma_id_list
                              | DETACH DELETE id
                              | DELETE comma_id_list
                              | DELETE id
                              | REMOVE dot_id_list
                              | REMOVE id LABEL label_id_list
                              | RETURN return_expression_list'''
    if len(p) == 3:
        p[0] = p[1]+' '+str(p[2])
    elif len(p)== 5:
        p[0] = p[1]+' '+p[2]+':'+str(p[4])
    else:
        p[0] = p[1]+' '+p[2]+' '+str(p[3])
def p_condition_expr(p):
    '''condition : boolean_expr
                 | cond_expr'''
    p[0] = p[1]

def p_condition_boolean_expr(p):
    '''boolean_expr : boolean_term
                     | NOT condition
                     | condition OR condition
                     | condition AND condition
                     | condition XOR condition
                     | cond_expr EQUALS cond_expr
                     | cond_expr LESS cond_expr 
                     | cond_expr GREATER cond_expr
                     | cond_expr CONTAINS term
                     | cond_expr ENDS WITH term
                     | cond_expr STARTS WITH term'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[0] = p[1]+' '+str(p[2])
    elif len(p) == 5:
        p[0] = p[1]+' '+p[2]+' '+p[3]+' '+p[4]
    else:
        if p[2] == 'EQUALS':
            p[0] = p[1]+' = '+ p[3]
        elif p[2] == 'LESS':
            p[0] = p[1]+' < '+ p[3]
        elif p[2] == 'GRATER':
            p[0] = p[1]+' > '+ p[3]
        else:
            p[0] = p[1]+' '+p[2]+' '+ p[3]

def p_boolean_term(p):
    '''boolean_term : TRUE
        | FALSE
        | boolean_expr'''
    p[0] = str(p[1])

def p_cond_expr(p):
    '''cond_expr : term
                 | id LABEL label_id_list
                 | id DOT id
                 | cond_expr PLUS cond_expr
                 | cond_expr MINUS cond_expr
                 | cond_expr MULTIPLY cond_expr
                 | cond_expr DIVIDE cond_expr'''
    if len(p) == 2:
        p[0] = str(p[1])
    else:
        if p[2] == 'PLUS':
            p[0] = p[1]+' + '+p[2]
        elif p[2] == 'MINUS':
            p[0] = p[1]+' - '+p[2]
        elif p[2] == 'MULTIPLY':
            p[0] = p[1]+' * '+p[2]
        elif p[2] == 'DIVIDE':
            p[0] = p[1]+' / '+p[2]
        elif p[2] == 'DOT':
            p[0] = str(p[1])+'.'+str(p[3])
        elif p[2] == 'LABEL':
            p[0] = p[1]+':'+p[3]  
                
def p_term_expr(p):
    '''term : cond_expr 
            | number
            | str'''
    p[0] = p[1]

def p_str_expr(p):
    '''str : p_list_string
         | id '''
    if p[1] in names.keys():
        p[0] = p[1]
    else:
        p[0]='\''+p[1]+'\''

def p_type_dot_list_id(p):
    '''
    dot_id_list : dot_id_list id DOT id
                | dot_id_list id
                | id DOT id
    '''
    if len(p) == 5:
        p[0] = str(p[1])+','+str(p[2])+'.'+str(p[4])
    elif len(p) == 3:
        p[0] = str(p[1])+','+str(p[2])
    else:
        p[0] = str(p[1])+'.'+str(p[3])

def p_statement_expr(p):
    '''statement : expr_list'''
    p[0] = p[1]

def p_expr_list(p):
    '''
    expr_list : expr_list expression
              | expression
    '''
    if len(p) == 3:
        p[1] = str(p[1])
        p[2] = str(p[2])
        if '[' not in p[1] and ']' not in p[2]:
            p[0] = p[1]+','+p[2]
        else:
            p[0] = p[1]+' '+p[2]
    else:
        p[0] = p[1]

def p_expression_relation(p):
    '''expression : RELATION relation_expr_list
                  | RELATION RIGHT relation_expr_list
                  | RELATION LEFT relation_expr_list
                  | RELATION RIGHT
                  | RELATION LEFT
                  | RELATION'''
    if len(p)==2:
        p[0] = '-[]-'
    elif len(p)==3:
        if p[2] == 'RIGHT':
            p[0] ='-[]->'
        elif p[2] == 'LEFT':
            p[0] = '<-[]-'
        else:
            p[0] = '-['+str(p[2])+']-'
    elif len(p)==4:
        if p[2] == 'RIGHT':
            p[0] ='-['+str(p[3])+']->'
        elif p[2] == 'LEFT':
            p[0] = '<-['+str(p[3])+']-'
        else:
            p[0] = '-['+str(p[3])+']-'
        

def p_relation_expr_list(p):
    '''
    relation_expr_list : relation_expr_list relation_expression
                       | relation_expression
    '''
    if len(p) == 3:
        p[0] = str(p[1])+p[2]
    else:
        p[0] = p[1]

def p_relation_expression(p):
    '''relation_expression : id
                           | TYPE type_id_list 
                           | MAP key_value_list
                           | LENGTH length_property
                           | LENGTH '''
    if len(p) == 2 and p[1] != 'LENGTH':
        p[0] = p[1]
        names[p[1]] = 'VARIABLE'
    elif p[1] == 'TYPE':
        p[0] = ':'+p[2]
    elif p[1] == 'LENGTH':
        if len(p) == 2:
            p[0] = '*'
        else:
            p[0] = '*'+p[2]
    else:
        p[0] = '{'+p[2]+'}'

def p_length_property(p):
    '''length_property : START number END number
                       | START number
                       | END number '''
    if len(p)==5:
        p[0] = str(p[2])+'..'+str(p[4])
    else:
        if p[1] == 'START':
            p[0] = str(p[2])+'..'
        else:
            p[0] = '..'+str(p[2])

def p_type_list_id(p):
    '''
    type_id_list : type_id_list id
                 | id
    '''
    if len(p) == 3:
        p[0] = str(p[1])+'|'+str(p[2]).upper()
    else:
        p[0] = str(p[1]).upper()

def p_comma_list_id(p):
    '''
    comma_id_list : comma_id_list id
                 | id
    '''
    if len(p) == 3:
        p[0] = str(p[1])+','+str(p[2])
    else:
        p[0] = str(p[1])

def p_expression_node(p):
    '''expression : NODE node_expr_list
                  | NODE '''
    if (len(p)==3):
        p[0] = '('+str(p[2])+')'
    else:
        p[0] = '()'
    

def p_node_expr_list(p):
    '''
    node_expr_list : node_expr_list node_expression
              | node_expression
    '''
    if len(p) == 3:
        p[0] = str(p[1])+p[2]
    else:
        p[0] = p[1]
      
def p_node_expression(p):
    '''node_expression : id
                | LABEL label_id_list 
                | MAP key_value_list'''
    
    if len(p)==2:
        p[0] = p[1]
        names[p[1]] = 'VARIABLE'
    elif p[1]== 'LABEL':
        p[0] = ':'+p[2]
    else:
        p[0] = '{'+p[2]+'}'

def p_key_value_list(p):
    '''key_value_list : key_value_list key_value_value
                    | key_value_value'''
    if len(p) == 3:
        p[0] = str(p[1])+','+p[2]
    else:
        p[0] = p[1]

def p_return_expression_list(p):
    """return_expression_list : return_expression_list return_expression_value
                              | return_expression_value"""
    if len(p) == 3:
        p[0] = str(p[1])+' '+p[2]
    else:
        p[0] = p[1]

def p_key_value_value(p):
    'key_value_value : KEY id VALUE p_list_string'
    if p[4] not in names.keys():
        p[0]=p[2]+':\''+p[4]+'\''
    else:
        p[0]=p[2]+':'+p[4]

def p_return_expression_value(p):
    """return_expression_value : dot_id_list
                               | comma_id_list
                               | id"""
    p[0] = p[1]

def p_expression_number(p):
    'number : NUMBER'
    p[0] = p[1]

def p_label_list_id(p):
    '''
    label_id_list : label_id_list id
              | id
    '''
    if len(p) == 3:
        p[0] = str(p[1])+':'+str(p[2]).lower().title()
    else:
        p[0] = str(p[1]).lower().title()

def p_list_string(p):
    '''
    p_list_string : p_list_string id
              | id
    '''
    if len(p) == 3:
        p[0] = str(p[1])+' '+str(p[2])
    else:
        p[0] = str(p[1])

def p_id(p):
    '''id : ID
          | FALSE
          | TRUE'''
    p[0] = str(p[1]).lower()


def p_error(p):
    if p == None:
        token = "end of file"
    else:
        token = f"{p.type}({p.value}) on line {p.lineno}"

    print(f"Syntax error: Unexpected {token}")

def main():
    
    yacc.yacc()
    
    model = SpeechRecognition.load_from_checkpoint("./speechtocypherdata/speech_recognition_model.pt")
    #uncomment this part to try recording a 10 second query, on macos you will need to install portaudio with brew
    """
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    sr.AudioFile('./output.wav')
    """
    predictions = model.predict(["output.wav"])
    print(predictions)
    
    res = yacc.parse(wordtodigits.convert(predictions[0]))
    print(res)
    
    dati = driver.cypherQuery(res)
    G = movie_graph_from_cypher(dati)
    pos = nx.spring_layout(G,k=0.5, iterations=20)
    nx.draw(G, pos, with_labels=True, font_size=8,  font_weight='bold', node_color='green', arrows=True)
    nx.draw_networkx_edge_labels(G,pos, edge_labels=dict([((n1, n2),type)
                    for n1, n2, type in G.edges.data('type_')]),font_size=6,font_color='r')
    plt.show()

if __name__ == "__main__":
    main()