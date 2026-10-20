import numpy.typing as npt
import numpy as np

ALGORITHM_REGISTRY = {}

def register_algorithm(name=None):
    def decorator(func):
        func_name = name or func.__name__
        if func_name in ALGORITHM_REGISTRY:
            raise ValueError(f"Function '{func_name}' is already registered.")
        ALGORITHM_REGISTRY[func_name] = func
        return func
    return decorator

@register_algorithm("bon")
def bon(final_answers: npt.NDArray[np.str_], 
        rewards: npt.NDArray[np.floating],
        **kwargs) -> int:
    return np.argmax(rewards)

@register_algorithm("sc")
def bon(final_answers: npt.NDArray[np.str_], 
        rewards: npt.NDArray[np.floating],
        **kwargs) -> int:
    unique_final_answers, counts = np.unique(final_answers, return_counts=True)
    most_frequent_answer = unique_final_answers[np.argmax(counts)]
    return np.where(final_answers == most_frequent_answer)[0][0]

@register_algorithm("wbon")
def wbon(final_answers: npt.NDArray[np.str_], 
         rewards: npt.NDArray[np.floating], 
         wbon_apply_sigmoid=False,
         **kwargs) -> int:
    if wbon_apply_sigmoid:
        rewards = 1 / (1 + np.exp(-rewards))
    unique_final_answers, indices = np.unique(final_answers, return_inverse=True)
    weighted_rewards = np.zeros(len(unique_final_answers))
    for i, reward in enumerate(rewards):
        weighted_rewards[indices[i]] += reward
    chosen_answer = unique_final_answers[np.argmax(weighted_rewards)]
    return np.where(final_answers == chosen_answer)[0][0]

def _bootstrap_bestofn_dist(final_answers: npt.NDArray[np.str_], 
                            rewards: npt.NDArray[np.floating], 
                            m: int, 
                            unique_final_answers=None):
    if unique_final_answers is None:
        unique_final_answers, counts = np.unique(final_answers, return_counts=True)
    dist = np.zeros(len(unique_final_answers))
    N = len(rewards)
    sorted_indices = np.argsort(rewards)
    sorted_indices_selctions_probs = ((np.arange(N) + 1) / N)**m - (np.arange(N) / N)**m
    for i in range(len(unique_final_answers)):
        final_ans = unique_final_answers[i]
        final_ans_sorted_indices = np.where((final_answers[sorted_indices] == final_ans))[0]
        dist[i] = np.sum(sorted_indices_selctions_probs[final_ans_sorted_indices])
    return dist, unique_final_answers

@register_algorithm("mob_poly_m")
def mob_poly_m(final_answers: npt.NDArray[np.str_], 
               rewards: npt.NDArray[np.floating],
               mob_alpha: float = 0.5,
               **kwargs) -> int:
    n = len(final_answers)
    m = max(int(n ** mob_alpha), 1)
    dist, unique_final_answers = _bootstrap_bestofn_dist(final_answers, rewards, m)
    chosen_answer = unique_final_answers[np.argmax(dist)]
    return np.where(final_answers == chosen_answer)[0][0]


@register_algorithm("mob_adaptive_m")
def mob_adaptive_m(final_answers: npt.NDArray[np.str_], 
                   rewards: npt.NDArray[np.floating], 
                   mob_q: float = 0.75,
                   return_m_value = False,
                   **kwargs) -> int:
    n = len(final_answers)
    if n == 1:
        return 0
    unique_final_answers, counts = np.unique(final_answers, return_counts=True)
    m_candidates = set()
    for i in range(50):
        m = max(1, int((mob_q ** i) * n))
        m_candidates.add(m)
        if m == 1:
            break
    m_candidates = sorted(list(m_candidates))
    dists = np.zeros((len(m_candidates), len(unique_final_answers)))
    diffs = np.zeros((len(m_candidates) - 1,))
    min_diff = np.inf
    min_diff_i = None
    for i, m in enumerate(m_candidates):
        dist, _ = _bootstrap_bestofn_dist(final_answers, rewards, m, unique_final_answers)
        dists[i] = dist
        if i > 0:
            diffs[i - 1] = np.sum(np.abs(dists[i] - dists[i - 1]))
            if diffs[i - 1] < min_diff:
                min_diff = diffs[i - 1]
                min_diff_i = i

    dist = dists[min_diff_i]
    chosen_answer = unique_final_answers[np.argmax(dist)]
    if not return_m_value:
        return np.where(final_answers == chosen_answer)[0][0]
    else:
        return np.where(final_answers == chosen_answer)[0][0], m_candidates[min_diff_i]