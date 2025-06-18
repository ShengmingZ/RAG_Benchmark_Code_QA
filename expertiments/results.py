from dataclasses import dataclass

@dataclass
class ExpResults:
    @dataclass
    class Preliminary:
        @dataclass
        class conala:
            single = 0.548
            oracle = 0.571

    @dataclass
    class Recall:
        @dataclass
        class conala:
            recall_100 = 0.571
            recall_80 = 0.512
            recall_60 = 0.488
            recall_40 = 0.464
            recall_20 = 0.44
            recall_0 = 0.429
