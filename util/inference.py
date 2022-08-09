import numpy as np
import re
import copy
from collections import OrderedDict
import pandas as pd
from sklearn.tree import _tree, plot_tree
from pysmt.shortcuts import Symbol, And, Or, Iff, Not, Bool, Solver, Implies
from pysmt.oracles import get_logic


def get_inference_vars(solution_list):
    var_c0 = [c for c in list(solution_list) if ("c0" in c[0])]
    var_c1 = [c for c in list(solution_list) if ("c1" in c[0])]
    var_v = [c for c in list(solution_list) if ("v" in c[0])]
    var_d0 = [c for c in list(solution_list) if ("d0" in c[0])]
    var_d1 = [c for c in list(solution_list) if ("d1" in c[0])]
    var_u = [c for c in list(solution_list) if ("u" in c[0])]

    return_dict = {
        "c0": var_c0,
        "c1": var_c1,
        "v": var_v,
        "d0": var_d0,
        "d1": var_d1,
        "u": var_u,
    }
    return return_dict


def var_size(vars_dict):
    pattern = re.compile(r"\((\d+)\)")
    nodes = [int(pattern.findall(c[0])[0]) for c in vars_dict["c0"]]
    num_nodes = len(nodes)
    num_features = len(vars_dict["d0"]) // num_nodes
    return num_nodes, num_features


def convert_2d(arr2d, num_nodes, num_features):
    arr_out = np.empty((num_features, num_nodes, 2), dtype="O")
    pattern = re.compile(r"\((\d+,\d+)\)")
    for c in arr2d:
        ij = [int(i) for i in pattern.findall(c[0])[0].split(",")]
        arr_out[ij[0] - 1, ij[1] - 1, 0] = Symbol(c[0].replace("'", ""))
        arr_out[ij[0] - 1, ij[1] - 1, 1] = c[1]
    return arr_out


def convert_1d(arr1d, num_var):
    arr_out = np.empty((num_var, 2), dtype="O")
    pattern = re.compile(r"\((\d+)\)")
    for c in arr1d:
        i = int(pattern.findall(c[0])[0])
        arr_out[i - 1, 0] = Symbol(c[0].replace("'", ""))
        arr_out[i - 1, 1] = c[1]
    return arr_out


def prediction_formula(inference_vars, features):
    inference_vars = get_inference_vars(inference_vars)
    N, K = var_size(inference_vars)
    arr_feat = np.array([Symbol(f"f({i+1})") for i in range(K)])
    ch0 = np.array([Symbol(f"ch0({i+1})") for i in range(N)])
    ch1 = np.array([Symbol(f"ch1({i+1})") for i in range(N)])

    c0 = convert_1d(inference_vars["c0"], N)
    c1 = convert_1d(inference_vars["c1"], N)
    v = convert_1d(inference_vars["v"], N)
    u = convert_2d(inference_vars["u"], N, K)
    d0 = convert_2d(inference_vars["d0"], N, K)
    d1 = convert_2d(inference_vars["d1"], N, K)

    all_assignments = []
    for i in range(K):
        all_assignments.append(Iff(arr_feat[i], Bool(features[i])))
    for i in range(N):
        all_assignments.append(Iff(c0[i, 0], Bool(c0[i, 1])))
        all_assignments.append(Iff(c1[i, 0], Bool(c1[i, 1])))
        all_assignments.append(Iff(v[i, 0], Bool(v[i, 1])))
    for i in range(K):
        for j in range(N):
            all_assignments.append(Iff(u[i, j, 0], Bool(u[i, j, 1])))
            all_assignments.append(Iff(d0[i, j, 0], Bool(d0[i, j, 1])))
            all_assignments.append(Iff(d1[i, j, 0], Bool(d1[i, j, 1])))

    formula = []
    for j in range(N):
        all_ands = []
        for r in range(K):
            left = Or(And(d0[r, j, 0], arr_feat[r]), Not(u[r, j, 0]))
            right = Or(And(d1[r, j, 0], Not(arr_feat[r])), Not(u[r, j, 0]))
            all_ands.append(Or(left, right))
        formula.append(Iff(And(And(all_ands), v[j, 0], c1[j, 0]), ch1[j]))
        formula.append(Iff(And(And(all_ands), v[j, 0], c0[j, 0]), ch0[j]))
        formula.append(Implies(ch0[j], Not(ch1[j])))
        formula.append(Implies(ch1[j], Not(ch0[j])))

    return formula, all_assignments


def robustness_formula(inference_vars, ch0_vals=None, ch1_vals=None):
    inference_vars = get_inference_vars(inference_vars)
    N, K = var_size(inference_vars)
    arr_feat = np.array([Symbol(f"f({i + 1})") for i in range(K)])
    ch0 = np.array([Symbol(f"ch0({i + 1})") for i in range(N)])
    ch1 = np.array([Symbol(f"ch1({i + 1})") for i in range(N)])

    c0 = convert_1d(inference_vars["c0"], N)
    c1 = convert_1d(inference_vars["c1"], N)
    v = convert_1d(inference_vars["v"], N)
    u = convert_2d(inference_vars["u"], N, K)
    d0 = convert_2d(inference_vars["d0"], N, K)
    d1 = convert_2d(inference_vars["d1"], N, K)

    ch_assignements = []
    if ch1_vals is not None:
        for chi in ch1_vals:
            ch_assignements.append(ch1[chi])
    if ch0_vals is not None:
        for chi in ch0_vals:
            ch_assignements.append(ch0[chi])

    all_assignments = []
    for i in range(N):
        all_assignments.append(Iff(c0[i, 0], Bool(c0[i, 1])))
        all_assignments.append(Iff(c1[i, 0], Bool(c1[i, 1])))
        all_assignments.append(Iff(v[i, 0], Bool(v[i, 1])))
    for i in range(K):
        for j in range(N):
            all_assignments.append(Iff(u[i, j, 0], Bool(u[i, j, 1])))
            all_assignments.append(Iff(d0[i, j, 0], Bool(d0[i, j, 1])))
            all_assignments.append(Iff(d1[i, j, 0], Bool(d1[i, j, 1])))

    formula = []
    for j in range(N):
        all_ands = []
        for r in range(K):
            left = Or(And(d0[r, j, 0], arr_feat[r]), Not(u[r, j, 0]))
            right = Or(And(d1[r, j, 0], Not(arr_feat[r])), Not(u[r, j, 0]))
            all_ands.append(Or(left, right))
        formula.append(Iff(And(And(all_ands), v[j, 0], c1[j, 0]), ch1[j]))
        formula.append(Iff(And(And(all_ands), v[j, 0], c0[j, 0]), ch0[j]))
        formula.append(Implies(ch0[j], Not(ch1[j])))
        formula.append(Implies(ch1[j], Not(ch0[j])))

    formula.append(Or(ch_assignements))

    return formula, all_assignments


def inference_sat(formula, all_assignments):
    target_logic = get_logic(And(formula))
    print("Target Logic: %s" % target_logic)
    solver = Solver(name="z3", logic=target_logic)
    solver.add_assertion(And(formula))
    solver.add_assertion(And(all_assignments))

    if solver.solve():
        model = solver.get_model()

        result = []
        for mi in model:
            result.append(mi)
        return result

    else:
        print("No solution found")
        return None


def create_graph(dt):
    tree_ = dt.tree_
    N = len(tree_.feature)

    graph = {str(i): [] for i in range(N)}

    left = tree_.children_left
    right = tree_.children_right

    for i in range(N):
        if left[i] != -1:
            graph[str(i)] = [str(left[i]), str(right[i])]
        else:
            graph[str(i)] = []

    return graph


def convert_tree_inds(dt):
    graph = create_graph(dt)

    visited = []  # List to keep track of visited nodes.
    queue = []  # Initialize a queue

    def bfs(visited, graph, node):
        visited.append(node)
        queue.append(node)

        while queue:
            s = queue.pop(0)

            for neighbour in graph[s]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
        return visited

    visited = bfs(visited, graph, "0")

    return {visited[i]: str(i) for i in range(len(visited))}


def create_tree_vars(dt):
    tree_ = dt.tree_
    N = len(tree_.feature)

    left = tree_.children_left
    right = tree_.children_right

    v = np.empty(N, dtype=bool)
    c0 = np.empty(N, dtype=bool)
    c1 = np.empty(N, dtype=bool)
    l = np.empty((N, N), dtype=bool)
    r = np.empty((N, N), dtype=bool)
    p = np.empty((N, N), dtype=bool)

    v.fill(False)
    c0.fill(False)
    c1.fill(False)
    l.fill(False)
    r.fill(False)
    p.fill(False)

    for i in range(N):
        if left[i] != -1:
            l[i, left[i]] = True
            r[i, right[i]] = True
            if right[i] > 0:
                p[right[i], i] = True
            if left[i] > 0:
                p[left[i], i] = True
        else:
            v[i] = True
            if i in left:
                left_i = np.where(left == i)[0]
                if left_i > 0:
                    p[i, left_i] = True
            else:
                right_i = np.where(right == i)[0]
                if right_i > 0:
                    p[i, right_i] = True

            if np.argmax(tree_.value[i]) == 0:
                c0[i] = True
            else:
                c1[i] = True

    return v, c0, c1, l, r, p


def create_tree_related(tree):
    v, c0, c1, l, r, p = create_tree_vars(tree)

    var_assignements = set()

    N = len(v)

    ind_conversion = convert_tree_inds(tree)

    for i in range(N):
        ind_i = int(ind_conversion[str(i)])
        var_assignements.add(Iff(Symbol(f"v({ind_i + 1})"), Bool(bool(v[i]))))
        var_assignements.add(Iff(Symbol(f"c0({ind_i + 1})"), Bool(bool(c0[i]))))
        var_assignements.add(Iff(Symbol(f"c1({ind_i + 1})"), Bool(bool(c1[i]))))

        for j in range(N):
            ind_j = int(ind_conversion[str(j)])
            var_assignements.add(
                Iff(Symbol(f"l({ind_i + 1},{ind_j + 1})"), Bool(bool(l[i, j])))
            )
            var_assignements.add(
                Iff(Symbol(f"r({ind_i + 1},{ind_j + 1})"), Bool(bool(r[i, j])))
            )
            var_assignements.add(
                Iff(Symbol(f"p({ind_i + 1},{ind_j + 1})"), Bool(bool(p[i, j])))
            )

    return var_assignements


def get_thresholds(dt):
    tree = dt.tree_
    dict_feat_thresh = {}
    for i in range(len(tree.feature)):
        if tree.feature[i] != _tree.TREE_UNDEFINED:
            if dict_feat_thresh.get(tree.feature[i], None) is None:
                dict_feat_thresh[tree.feature[i]] = [tree.threshold[i]]
            else:
                dict_feat_thresh[tree.feature[i]].append(tree.threshold[i])
    for k, v in dict_feat_thresh.items():
        dict_feat_thresh[k] = list(set(v))
    return dict_feat_thresh


def disc_data(data, dict_thresh):
    used_inds = list(dict_thresh.keys())
    all_inds = list(range(data.shape[1]))
    not_used_inds = list(set(all_inds).difference(set(used_inds)))

    num_cols = np.sum([len(v) for k, v in dict_thresh.items()])
    col_names = [f"{k}_{v[i]}" for k, v in dict_thresh.items() for i in range(len(v))]
    data_ = pd.DataFrame(columns=col_names)

    for k, v in dict_thresh.items():
        for vi in v:
            col = f"{k}_{vi}"
            data_[col] = data[:, k] > vi

    return data_


def get_lineage(tree):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = tree.tree_.feature

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = "l"
        else:
            parent = np.where(right == child)[0].item()
            split = "r"

        lineage.append([parent, split, threshold[parent], features[parent]])

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]
    all_paths = []
    for child in idx:
        path = []
        for node in recurse(left, right, child):
            path.append(node)
        all_paths.append(path)

    return all_paths


def find_boundaries(dt):
    lineages = get_lineage(dt)
    ind_conversion = convert_tree_inds(dt)

    box_dict = {}

    for lineage in lineages:
        box_fr = list(set([li[-1] for li in lineage[:-1]]))
        box = {str(bi): [-np.inf, np.inf] for bi in box_fr}
        for li in lineage[:-1]:
            if li[1] == "l":
                box[str(li[-1])][1] = min(box[str(li[-1])][1], li[2])
            else:
                box[str(li[-1])][0] = max(box[str(li[-1])][0], li[2])
        box_dict[f"{int(ind_conversion[str(lineage[-1])]) + 1}"] = box
    return box_dict


def create_fr_related(tree, column_names):
    lineage = get_lineage(tree)

    a = set()
    u = set()

    ind_conversion = convert_tree_inds(tree)

    for paths in lineage:
        leaf = paths[-1]
        leaf_conv = int(ind_conversion[str(leaf)])
        for path_i in paths[:-1]:
            col_name = f"{path_i[-1]}_{path_i[-2]}"
            col_num = np.where(col_name == column_names)[0][0]

            path_i_conv = int(ind_conversion[str(path_i[0])])

            a.add(Symbol(f"a({col_num + 1},{path_i_conv + 1})"))
            u.add(Symbol(f"u({col_num + 1},{leaf_conv + 1})"))
            a.add(Not(Symbol(f"a({col_num + 1},{leaf_conv + 1})")))

    return a, u


def path_to_leaves(children_left, children_right):
    """
    Given a tree, find the path between the root node and all the leaves
    """
    leaf_paths = OrderedDict()
    path = []

    def _find_leaves(root, path, depth, branch):
        children = [children_left[root], children_right[root]]
        children = [c for c in children if c != -1]

        if len(path) > depth:
            path[depth] = (root, branch)
        else:
            path.append((root, branch))

        if len(children) == 0:
            nodes, dirs = zip(*path[: depth + 1])
            leaf_paths[root] = list(zip(nodes[:-1], dirs[1:]))
        else:
            _find_leaves(children_left[root], path, depth + 1, "left")
            _find_leaves(children_right[root], path, depth + 1, "right")

    _find_leaves(0, path, 0, "root")
    return leaf_paths


def leaf_boxes(tree):
    n_nodes = len(tree.feature)
    n_features = tree.n_features

    children_left = tree.children_left
    children_right = tree.children_right

    feature = tree.feature
    threshold = tree.threshold
    value = tree.value

    leaf_paths = path_to_leaves(children_left, children_right)
    leaves = sorted(leaf_paths)

    n_leaves = len(leaves)

    # Make this sparse in the future, if necessary
    max_corners = np.ones((n_leaves, n_features)) * np.inf
    min_corners = np.ones((n_leaves, n_features)) * -np.inf

    for i, leaf in enumerate(leaves):
        path = leaf_paths[leaf]
        for node, step in path:
            feat_id = feature[node]
            thresh = threshold[node]

            # left, x <= t; r=t, l=-inf
            # right, x > t; r=inf, l=t

            if step == "left":
                max_corners[i, feat_id] = thresh
            else:
                min_corners[i, feat_id] = thresh

    return leaves, min_corners, max_corners


def box_intersection(mini, minj, maxi, maxj):
    maxm = np.maximum(mini[:, :, None], minj.T[None, :, :])
    minm = np.minimum(maxi[:, :, None], maxj.T[None, :, :])
    # It looks like min > max, but these are for u, l
    check = np.all(((minm - maxm) > 0), axis=1)

    return check
