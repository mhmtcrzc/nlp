import ZemberekFunctions as ze
import sys


class Preprocess:

    def __init__(self, host):
        super().__init__()
        self.instance = ze.ZemberekFunctions(host)

    def preprocess(self, text):
        try:
            tokens = self.instance.tokenize(text)
        except:
            print("Connection error, please check the connection status. \nExiting the program...")
            sys.exit()
        stemming = []
        for t in tokens:
            if t.type == 'Word':
                root = self.instance.analyze(t.token).results[0].best.lemmas[0]
                if not self.instance.is_stop_word(root):
                    stemming.append(str(root))
        return stemming



