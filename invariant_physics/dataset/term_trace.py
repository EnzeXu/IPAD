import matplotlib.pyplot as plt
from ._utils import extract, simplify_and_replace_constants


class TermTrace:
    def __init__(self, num_env):
        self.num_env = num_env
        self.class_count = 0
        self.term_list = [[] for _ in range(self.num_env)]
        self.class_list = [[] for _ in range(self.num_env)]
        self.term_dic = dict()

    def add_iteration_result(self, origin_terms):
        # print(f"origin_terms={origin_terms}")
        for i, one_env_term in enumerate(origin_terms):
            # print(f"one_env_term={one_env_term}")
            full_terms, terms, coefficient_terms = extract(one_env_term)
            terms = [simplify_and_replace_constants(item) for item in terms]
            self.term_list[i].append(terms)
            terms_string = str(terms)
            if terms_string not in self.term_dic:
                self.class_count += 1
                self.term_dic[terms_string] = self.class_count
                self.class_list[i].append(self.class_count)
            else:
                tmp_class = self.term_dic[terms_string]
                self.class_list[i].append(tmp_class)

    def draw_term_trace(self, save_path):
        plt.figure(figsize=(12, 6))
        for i in range(self.num_env):
            t = list(range(len(self.class_list[i])))
            plt.plot(t, self.class_list[i], linewidth=2, label=f"Env {i+1}", alpha=0.5)
        y_ticks = list(range(1, self.class_count + 1))
        y_tick_labels = list(self.term_dic.keys())  # Define the corresponding labels
        plt.yticks(y_ticks, y_tick_labels, fontsize=5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    tt = TermTrace(5)
    tt.add_iteration_result(['0.017917159594662988*x**2 - 0.36353854760004814*x*y + 2.9274559414338611*y', '0.019733015478124605*x**2 - 0.45856765133135316*x*y + 3.5282381551104285*y', '0.01467491829878713*x**2 - 0.4675662653334395*x*y + 3.653356467208934*y', '0.02038723964669689*x**2 - 0.5993416945030791*x*y + 4.0238759455446249*y', '0.019782887866451062*x**2 - 0.46303698153478301*x*y + 2.4564074205200801*y'])
    tt.add_iteration_result(['0.017917159594662988*x**2 - 0.36353854760004814*x*y + 2.9274559414338611*y',
                             '0.019733015478124605*x**2 - 0.45856765133135316*x*y + 3.5282381551104285*y',
                             '0.01467491829878713*x**2 - 0.4675662653334395*x*y + 3.653356467208934*y',
                             '0.02038723964669689*x**2 - 0.5993416945030791*x*y + 4.0238759455446249*y',
                             '0.019782887866451062*x**2 - 0.46303698153478301*x*y + 2.4564074205200801*y'])

    tt.add_iteration_result(['-0.30000487338808431*x*y + 1.0000066389264906*x + 1.9344303397627034e-5*y**2', '-0.38999981996970192*x*y + 1.1999994722553353*x - 7.2357716637977201e-7*y**2', '-0.41999978260429265*x*y + 1.2999990623118605*x - 5.0385314745347093e-7*y**2', '-0.51000017246528334*x*y + 1.1000002181836235*x + 8.0295203648603066e-7*y**2', '-0.3899894718566138*x*y + 0.8999477135018839*x - 8.671977334282311e-6*y**2'])

    tt.draw_term_trace("test/test_term_trace.png")


