class Base:
    def report(self):
        raise NotImplementedError("should be implemented in subclass")

    def report_latex(self):
        raise NotImplementedError("should be implemented in subclass")
