import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames


class SAM2VideoPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # Whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # Whether to clear non-conditioning memory of surrounding frames after adding correction clicks
        clear_non_cond_mem_around_input=False,
        # Whether to append frames with correction clicks to the conditioning frame list
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # Device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """Load a pretrained model from the Hugging Face hub."""
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["frames_tracked_per_obj"][obj_idx] = {}
        return obj_idx

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points or box to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if box is not None:
            if not clear_old_points:
                raise ValueError("cannot add box without clearing old points")
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device).reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            scale = self.image_size / torch.tensor([video_W, video_H], device=points.device)
            points = points * scale
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        reverse = False if is_init_cond_frame else obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].to(inference_state["device"], non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Use `add_new_points_or_box` instead."""
        warnings.warn("`add_new_points` is deprecated. Use `add_new_points_or_box` instead.")
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None].float().to(inference_state["device"])

        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = F.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        reverse = False if is_init_cond_frame else obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """Resize object scores to original video resolution and apply non-overlapping constraints."""
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] != (video_H, video_W):
            video_res_masks = F.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="nearest",
            )
        else:
            video_res_masks = any_res_masks
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        consolidate_at_video_res=False,
    ):
        """Consolidate per-object temporary outputs into a single output for all objects."""
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        consolidated_H = inference_state["video_height"] if consolidate_at_video_res else self.image_size // 4
        consolidated_W = inference_state["video_width"] if consolidate_at_video_res else self.image_size // 4
        consolidated_mask_key = "pred_masks_video_res" if consolidate_at_video_res else "pred_masks"

        consolidated_out = {
            consolidated_mask_key: torch.full(
                (batch_size, 1, consolidated_H, consolidated_W),
                NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
                if out is None:
                    out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            if out is None:
                continue
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] != consolidated_pred_masks.shape[-2:]:
                resized_obj_mask = F.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="nearest",
                )
                consolidated_pred_masks[obj_idx:obj_idx + 1] = resized_obj_mask
            else:
                consolidated_pred_masks[obj_idx:obj_idx + 1] = obj_mask

        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError("No input points or masks provided; please add inputs first.")

        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    if out["maskmem_features"] is None:
                        high_res_masks = F.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc
                    obj_output_dict[storage_key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
                obj_temp_output_dict[storage_key].clear()

            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(f"No inputs provided for object id {obj_id}.")
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        if start_frame_idx is None:
            start_frame_idx = min(
                t for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1) if start_frame_idx > 0 else []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    pred_masks = current_out["pred_masks"].to(inference_state["device"], non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                pred_masks_per_obj[obj_idx] = pred_masks

            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0) if len(pred_masks_per_obj) > 1 else pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            yield frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        temp_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in inference_state["temp_output_dict_per_obj"].values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or masks in all frames."""
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["frames_tracked_per_obj"].values():
            v.clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": [feat.expand(batch_size, -1, -1, -1) for feat in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [pos.expand(batch_size, -1, -1, -1) for pos in backbone_out["vision_pos_enc"]],
        }
        features = self._prepare_backbone_features(expanded_backbone_out)
        return (expanded_image,) + features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16).to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]

        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Run the memory encoder on high_res_masks."""
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16).to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, {"maskmem_pos_enc": maskmem_pos_enc})
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """Cache and retrieve maskmem_pos_enc as a constant."""
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """Remove an object id from the tracking state."""
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        if old_obj_idx_to_rm is None:
            if strict:
                raise RuntimeError(f"Cannot remove object id {obj_id} as it doesn't exist.")
            return inference_state["obj_ids"], updated_frames

        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        obj_input_frames_inds = set()
        obj_input_frames_inds.update(inference_state["point_inputs_per_obj"][old_obj_idx_to_rm])
        obj_input_frames_inds.update(inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm])
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(inference_state, frame_idx, obj_id, need_output=False)

        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = [i for i in old_obj_inds if i != old_obj_idx_to_rm]
        new_obj_ids = [old_obj_ids[i] for i in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        def _map_keys(container):
            new_kvs = [(old_idx_to_new_idx[k], container.pop(k)) for k in old_obj_inds if k in old_idx_to_new_idx]
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])
        _map_keys(inference_state["frames_tracked_per_obj"])

        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
        """Remove non-conditioning memory around the input frame for a specific object."""
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        non_cond_frame_outputs = inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """Remove non-conditioning memory around the input frame for all objects."""
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            self._clear_obj_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
