# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector

import torch.nn.functional as F
import math
from torchvision.ops import nms


@MODELS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                    bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_outs = self.roi_head.forward(x, rpn_results_list)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        #     rpn_data_samples = copy.deepcopy(batch_data_samples)
        #     # set cat_id of gt_labels to 0 in RPN
        #     for data_sample in rpn_data_samples:
        #         data_sample.gt_instances.labels = \
        #             torch.zeros_like(data_sample.gt_instances.labels)
        #
        #     rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
        #         x, rpn_data_samples, proposal_cfg=proposal_cfg)
        #     # avoid get same name with roi_head loss
        #     keys = rpn_losses.keys()
        #     for key in keys:
        #         if 'loss' in key and 'rpn' not in key:
        #             rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        #     losses.update(rpn_losses)
        # else:
        #     # TODO: Not support currently, should have a check at Fast R-CNN
        #     assert batch_data_samples[0].get('proposals', None) is not None
        #     # use pre-defined proposals in InstanceData for the second stage
        #     # to extract ROI features.
        #     rpn_results_list = [
        #         data_sample.proposals for data_sample in batch_data_samples
        #     ]

        rpn_data_samples = copy.deepcopy(batch_data_samples)
        _rpn_results_list, rpn_losses = self.rpn_meg(x, rpn_data_samples)
        losses.update(rpn_losses)

        from mmengine.data import InstanceData
        img_1 = InstanceData()
        img_1.bboxes = _rpn_results_list[:2000, 1:]
        img_1.scores = torch.ones_like(img_1.bboxes[:, 0])
        img_1.labels = torch.zeros_like(img_1.bboxes[:, 0])
        img_2 = InstanceData()
        img_2.bboxes = _rpn_results_list[2000:, 1:]
        img_2.scores = torch.ones_like(img_2.bboxes[:, 0])
        img_2.labels = torch.zeros_like(img_2.bboxes[:, 0])
        rpn_results_list = [img_1, img_2]

        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

    def rpn_meg(self, features, rpn_data_samples):
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_head.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_head.rpn_cls(t))
            pred_bbox_offsets_list.append(self.rpn_head.rpn_reg(t))

        featmap_sizes = [featmap.size()[-2:] for featmap in pred_cls_score_list]
        device = pred_cls_score_list[0].device
        batch_img_metas = [sample.metainfo for sample in rpn_data_samples]
        scale_factor = [max(metas['scale_factor']) for metas in batch_img_metas]
        all_anchors_list, _ = self.rpn_head.get_anchors(featmap_sizes, batch_img_metas, device=device)
        rpn_rois = find_top_rpn_proposals(
            pred_bbox_offsets_list[::-1], pred_cls_score_list[::-1], all_anchors_list[0][::-1], scale_factor)
        rpn_rois = rpn_rois.type_as(features[0])

        im_info = []
        _boxes = []
        for sample in rpn_data_samples:
            im_info.append(sample.gt_instances.bboxes.shape[0])
            _boxes.append(sample.gt_instances.bboxes)
        boxes = torch.zeros([2, 500, 4], dtype=torch.float32).to(_boxes[0].device)
        boxes[0, :im_info[0], :4] = _boxes[0]
        boxes[1, :im_info[1], :4] = _boxes[1]
        rpn_labels, rpn_bbox_targets = fpn_anchor_target(boxes, im_info, all_anchors_list[0][::-1])
        pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(pred_cls_score_list[::-1], pred_bbox_offsets_list[::-1])

        valid_masks = rpn_labels >= 0
        objectness_loss = softmax_loss(pred_cls_score[valid_masks], rpn_labels[valid_masks])
        pos_masks = rpn_labels == 0  # changed by yu >
        localization_loss = smooth_l1_loss(pred_bbox_offsets[pos_masks], rpn_bbox_targets[pos_masks], 1)  # config.rpn_smooth_l1_beta
        normalizer = 1 / valid_masks.sum().item()
        loss_rpn_cls = objectness_loss.sum() * normalizer
        loss_rpn_loc = localization_loss.sum() * normalizer
        loss_dict = {}
        loss_dict['loss_rpn_cls'] = loss_rpn_cls
        loss_dict['loss_rpn_loc'] = loss_rpn_loc
        return rpn_rois, loss_dict


    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=True)
        # connvert to DetDataSample
        results_list = self.convert_to_datasample(results_list)

        return results_list


@torch.no_grad()
def find_top_rpn_proposals(rpn_bbox_offsets_list, rpn_cls_prob_list, all_anchors_list, scale_factor):
    prev_nms_top_n = 12000
    post_nms_top_n = 2000
    batch_per_gpu = 2
    nms_threshold = 0.7
    box_min_size = 2
    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_probs_list = []
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid].permute(1, 2, 0).reshape(-1, 4)
            all_anchors = all_anchors_list[l]
            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            probs = rpn_cls_prob_list[l][bid].permute(1, 2, 0).reshape(-1, 2)
            temp = copy.deepcopy(probs[:, 0])
            probs[:, 0] = copy.deepcopy(probs[:, 1])
            probs[:, 1] = copy.deepcopy(temp)
            probs = torch.softmax(probs, dim=-1)[:, 1]  # 1
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_probs_list.append(probs)
        batch_proposals = torch.cat(batch_proposals_list, dim=0)
        batch_probs = torch.cat(batch_probs_list, dim=0)
        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(batch_proposals, box_min_size * scale_factor[bid])
        batch_proposals = batch_proposals[batch_keep_mask]
        batch_probs = batch_probs[batch_keep_mask]
        # prev_nms_top_n
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx]
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        keep = keep[:post_nms_top_n]
        batch_proposals = batch_proposals[keep]
        # cons the rois
        batch_inds = torch.ones(batch_proposals.shape[0], 1).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1)
        return_rois.append(batch_rois)

    if batch_per_gpu == 1:
        return batch_rois
    else:
        concated_rois = torch.cat(return_rois, axis=0)
        return concated_rois


def bbox_transform_inv_opr(bbox, deltas):
    max_delta = math.log(1000.0 / 16)
    """ Transforms the learned deltas to the final bbox coordinates, the axis is 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
    pred_ctr_x = bbox_ctr_x + deltas[:, 0] * bbox_width
    pred_ctr_y = bbox_ctr_y + deltas[:, 1] * bbox_height

    dw = deltas[:, 2]
    dh = deltas[:, 3]
    dw = torch.clamp(dw, max=max_delta)
    dh = torch.clamp(dh, max=max_delta)
    pred_width = bbox_width * torch.exp(dw)
    pred_height = bbox_height * torch.exp(dh)

    pred_x1 = pred_ctr_x - 0.5 * pred_width
    pred_y1 = pred_ctr_y - 0.5 * pred_height
    pred_x2 = pred_ctr_x + 0.5 * pred_width
    pred_y2 = pred_ctr_y + 0.5 * pred_height
    pred_boxes = torch.cat(
        (pred_x1.reshape(-1, 1), pred_y1.reshape(-1, 1),
         pred_x2.reshape(-1, 1), pred_y2.reshape(-1, 1)), dim=1)
    return pred_boxes


def filter_boxes_opr(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = (ws >= min_size) * (hs >= min_size)
    return keep


@torch.no_grad()
def fpn_anchor_target(boxes, im_info, all_anchors_list):
    final_labels_list = []
    final_bbox_targets_list = []
    for bid in range(2):  # config.train_batch_per_gpu
        batch_labels_list = []
        batch_bbox_targets_list = []
        for i in range(len(all_anchors_list)):
            anchors_perlvl = all_anchors_list[i]
            rpn_labels_perlvl, rpn_bbox_targets_perlvl = fpn_anchor_target_opr_core_impl(
                boxes[bid], im_info[bid], anchors_perlvl)
            batch_labels_list.append(rpn_labels_perlvl)
            batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)
        # here we samples the rpn_labels
        concated_batch_labels = torch.cat(batch_labels_list, dim=0)
        concated_batch_bbox_targets = torch.cat(batch_bbox_targets_list, dim=0)
        # sample labels
        pos_idx, neg_idx = subsample_labels(concated_batch_labels, 256, 0.5)  # config.num_sample_anchors, config.positive_anchor_ratio
        concated_batch_labels.fill_(-1)
        concated_batch_labels[pos_idx] = 0  # 1
        concated_batch_labels[neg_idx] = 1  # 0

        final_labels_list.append(concated_batch_labels)
        final_bbox_targets_list.append(concated_batch_bbox_targets)
    final_labels = torch.cat(final_labels_list, dim=0)
    final_bbox_targets = torch.cat(final_bbox_targets_list, dim=0)
    return final_labels, final_bbox_targets


def fpn_anchor_target_opr_core_impl(gt_boxes, im_info, anchors, allow_low_quality_matches=True):
    ignore_label = -1  # config.ignore_label
    # get the gt boxes
    valid_gt_boxes = gt_boxes[:int(im_info), :]
    valid_gt_boxes = valid_gt_boxes[valid_gt_boxes[:, -1].gt(0)]
    # compute the iou matrix
    anchors = anchors.type_as(valid_gt_boxes)
    overlaps = box_overlap_opr(anchors, valid_gt_boxes[:, :4])
    # match the dtboxes
    max_overlaps, argmax_overlaps = torch.max(overlaps, axis=1)
    gt_argmax_overlaps = my_gt_argmax(overlaps)
    del overlaps
    # all ignore
    labels = torch.ones(anchors.shape[0], device=gt_boxes.device, dtype=torch.long) * ignore_label
    # set negative ones
    labels = labels * (max_overlaps >= 0.3)  # config.rpn_negative_overlap
    # set positive ones
    fg_mask = (max_overlaps >= 0.7)  # config.rpn_positive_overlap
    if allow_low_quality_matches:
        gt_id = torch.arange(valid_gt_boxes.shape[0]).type_as(argmax_overlaps)
        argmax_overlaps[gt_argmax_overlaps] = gt_id
        max_overlaps[gt_argmax_overlaps] = 1
        fg_mask = (max_overlaps >= 0.7)  # config.rpn_positive_overlap
    # set positive ones
    fg_mask_ind = torch.nonzero(fg_mask, as_tuple=False).flatten()
    labels[fg_mask_ind] = 1
    # bbox targets
    bbox_targets = bbox_transform_opr(anchors, valid_gt_boxes[argmax_overlaps, :4])
    return labels, bbox_targets


def box_overlap_opr(box, gt):
    assert box.ndim == 2
    assert gt.ndim == 2
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    width_height = torch.min(box[:, None, 2:], gt[:, 2:]) - torch.max(
        box[:, None, :2], gt[:, :2]) + 1  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area_box[:, None] + area_gt - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def my_gt_argmax(overlaps):
    gt_max_overlaps, _ = torch.max(overlaps, axis=0)
    gt_max_mask = overlaps == gt_max_overlaps
    gt_argmax_overlaps = []
    for i in range(overlaps.shape[-1]):
        gt_max_inds = torch.nonzero(gt_max_mask[:, i], as_tuple=False).flatten()
        gt_max_ind = gt_max_inds[torch.randperm(gt_max_inds.numel(), device=gt_max_inds.device)[0,None]]
        gt_argmax_overlaps.append(gt_max_ind)
    gt_argmax_overlaps = torch.cat(gt_argmax_overlaps)
    return gt_argmax_overlaps


def bbox_transform_opr(bbox, gt):
    """ Transform the bounding box and ground truth to the loss targets.
    The 4 box coordinates are in axis 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height

    gt_width = gt[:, 2] - gt[:, 0] + 1
    gt_height = gt[:, 3] - gt[:, 1] + 1
    gt_ctr_x = gt[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
    target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
    target_dw = torch.log(gt_width / bbox_width)
    target_dh = torch.log(gt_height / bbox_height)
    target = torch.cat((target_dx.reshape(-1, 1), target_dy.reshape(-1, 1),
                        target_dw.reshape(-1, 1), target_dh.reshape(-1, 1)), dim=1)
    return target


def subsample_labels(labels, num_samples, positive_fraction):
    positive = torch.nonzero((labels != -1) & (labels != 0), as_tuple=False).squeeze(1)
    negative = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def fpn_rpn_reshape(pred_cls_score_list, pred_bbox_offsets_list):
    final_pred_bbox_offsets_list = []
    final_pred_cls_score_list = []
    for bid in range(2):  # config.train_batch_per_gpu
        batch_pred_bbox_offsets_list = []
        batch_pred_cls_score_list = []
        for i in range(len(pred_cls_score_list)):
            pred_cls_score_perlvl = pred_cls_score_list[i][bid] \
                .permute(1, 2, 0).reshape(-1, 2)
            pred_bbox_offsets_perlvl = pred_bbox_offsets_list[i][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            batch_pred_cls_score_list.append(pred_cls_score_perlvl)
            batch_pred_bbox_offsets_list.append(pred_bbox_offsets_perlvl)
        batch_pred_cls_score = torch.cat(batch_pred_cls_score_list, dim=0)
        batch_pred_bbox_offsets = torch.cat(batch_pred_bbox_offsets_list, dim=0)
        final_pred_cls_score_list.append(batch_pred_cls_score)
        final_pred_bbox_offsets_list.append(batch_pred_bbox_offsets)
    final_pred_cls_score = torch.cat(final_pred_cls_score_list, dim=0)
    final_pred_bbox_offsets = torch.cat(final_pred_bbox_offsets_list, dim=0)
    return final_pred_cls_score, final_pred_bbox_offsets


def softmax_loss(score, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], 2, device=score.device)  # config.num_classes,
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss


def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)
