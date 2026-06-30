def list_to_code_string(code_a):
    code_str = ' '.join(code_a)
    code_str = code_str.replace(' { ', ' {\n    ')  
    code_str = code_str.replace(' } ', '\n}\n')     
    code_str = code_str.replace(' ; ', ';\n    ')  
    code_str = code_str.replace(' \n ', '\n')     
    return code_str.strip()  

code_a = ['int', 'main', '(', ')', '{', 
          'int', 'x', '=', '10', ';', 
          'int', 'y', '=', '20', ';', 
          'printf', '(', '"%d\\n"', ',', 'x', '+', 'y', ')', ';', 
          '}']

code_string = list_to_code_string(code_a)
print(code_string)