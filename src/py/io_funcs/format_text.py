import os

def format_text(input_str):
    ### upper case to lower case ###
    output_str = input_str.lower()
    
    fp = open('temp','w')
    fp.write(output_str+'\n')
    fp.close()
    
    ### remove punctuation marks ###
    os.system('sed -i \'s/\([[:punct:]]\)//g\' temp')

    ### remove trailing spaces ###
    os.system('sed -i -e \'s/^[ \t]*//;s/[ \t]*$//\' temp')

    ### remove multiple spaces ###
    os.system('sed -i \'s/  */ /g\' temp')
    
    fp = open('temp','r')
    for i in fp.readlines():
        output_str = i.strip()
    fp.close()

    return output_str

if __name__ == "__main__":
    
    input_str = 'this kid  is Too much fun!!'
    print format_text(input_str)