import torch
from scipy.optimize import linear_sum_assignment


class HungarianLoss:
    def __init__(self, lambda_iou, lambda_l1, last_hidden_size, num_labels):
        self.l_iou = lambda_iou
        self.l_l1 = lambda_l1
        self.last_hidden_size = last_hidden_size
        self.num_labels = num_labels

        self._box_losses = dict()

    @staticmethod
    def giou_1d(y_pred, y_target):
        """
        Generalized Intersection Over Union for 1D "Boxes"
        """
        p_c = y_pred["center"]
        p_l = y_pred["len"] / 2
        t_c = y_target[1]
        t_l = y_target[2] / 2

        # Prediction start and end
        p_s = p_c - p_l
        p_e = p_c + p_l

        # Target start and end
        t_s = t_c - t_l
        t_e = t_c + t_l

        # Convex start and end
        c_s = min(p_s, t_s)
        c_e = max(p_e, t_e)

        # Intersection start and end
        i_s = max(p_s, t_s)
        i_e = min(p_e, t_e)

        intersection = max(0, i_e - i_s)
        union = p_e - p_s + t_e - t_s - intersection

        convex = c_s - c_e

        return intersection / union - (convex - union) / convex

    def _is_backgound(self, y_target):
        return y_target[0] == self.num_labels

    @staticmethod
    def _get_logit(pred, target):
        return pred["class"][int(target[0])]

    def _compute_box_loss(self, pred, target, id_pred, id_target):
        if target[1] > 1 or target[2] > 1:
            raise RuntimeError(
                "Trying to calculate box loss on a non normalized target"
            )

        self._box_losses[(id_pred, id_target)] = self.l_iou * self.giou_1d(
            pred, target
        ) + self.l_l1 * torch.abs(pred["center"] - target[1])

        return self._box_loss(id_pred, id_target)

    def _box_loss(self, id_pred, id_target):
        return self._box_losses.get((id_pred, id_target), 0)

    def _class_loss(self, pred, target):
        return torch.log(self._get_logit(pred, target))

    def _matching_phase(self, y_pred, y_target):
        # Computation of cost matrix for hungarian algorithm
        cost_matrix = torch.zeros((self.last_hidden_size, self.last_hidden_size))
        self._box_losses = dict()

        for i in range(self.last_hidden_size):
            for j in range(self.last_hidden_size):
                cost = 0
                if not self._is_backgound(y_target[j]):
                    cost = -self._get_logit(
                        y_pred[i], y_target[j]
                    ) + self._compute_box_loss(y_pred[i], y_target[j], i, j)

                cost_matrix[i, j] = cost

        # It can be improved with better libraries
        id_pred, id_target = linear_sum_assignment(cost_matrix)

        return zip(id_pred, id_target)

    def __call__(self, y_pred, y_target):
        """
        The network will always generate N object to classify, where N is the dimension
        of the last hidden size of the transformers (768). For each one will generate a probability
        to belong to each class.
        Assumptio to receive the pred and target as dict in the following form:
        y_pred[i] = {'class' : [s1, .. , si, ..., sn],
                'center': c,
                'len'   : l}

        y_target[i] = [s', c', l']

        c, c'. l and l' are normalized wrt the length of the sequence, i.e. they are in the range [0, 1]

        """
        l = torch.zeros(1)
        extend = torch.full((self.last_hidden_size - len(y_target), 3), self.num_labels)
        y_target = torch.vstack((y_target, extend))
        opt_pairs = self._matching_phase(y_pred, y_target)

        for id_pred, id_target in opt_pairs:
            l += -self._class_loss(
                y_pred[id_pred], y_target[id_target]
            ) + self._box_loss(id_pred, id_target)

        return l