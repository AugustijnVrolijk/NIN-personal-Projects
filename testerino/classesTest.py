


class MainClass():
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
        self.doSmth()
        pass

    def doSmth(self):
        result = self.val1 + self.val2
        print(f"result main: {result}")



class inheritClass(MainClass):
    def __init__(self, val1, val2):
        super().__init__(val1, val2)
        self.doSmth()

    def doSmth(self):
        
        result = self.val1 - self.val2
        print(f"result inherited: {result}")

def main():
    val1 = 120
    val2 = 55
    tester = inheritClass(val1, val2)

if __name__ == "__main__":
    main()