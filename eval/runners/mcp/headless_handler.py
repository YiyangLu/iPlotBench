"""Headless handler for server-side screenshot generation.

Uses Kaleido for generating screenshots without a browser.
"""

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from .context import get_current_session
from .plot_store import get_plot_store


class HeadlessHandler:
    """Handle plot commands and generate screenshots server-side."""

    def __init__(self):
        self.screenshot_width = 800
        self.screenshot_height = 600
        self.screenshot_scale = 2  # 2x for retina

    def handle_init(self, session_id: str, plot_id: int) -> dict:
        """Handle initial plot creation.

        Args:
            session_id: Session identifier
            plot_id: Plot ID

        Returns:
            {"success": True, "interaction_id": int}
        """
        store, _ = get_current_session()
        if not store:
            return {"success": False, "error": "No session"}

        # Load plot JSON
        plot_store = get_plot_store()
        fig_json = plot_store.get_plot(session_id, plot_id)
        if not fig_json:
            return {"success": False, "error": f"Plot {plot_id} not found"}

        # Log init interaction
        interaction_id = store.log_interaction(
            session_id=session_id,
            plot_id=plot_id,
            event_type="init",
            payload={"plot_id": plot_id},
        )

        # Generate screenshot
        screenshot_path, screenshot_size = self._generate_screenshot(
            session_id, plot_id, interaction_id, fig_json
        )

        # Update interaction with screenshot
        if screenshot_path:
            store.update_interaction_screenshot(
                session_id, interaction_id, screenshot_path, screenshot_size
            )

        return {"success": True, "interaction_id": interaction_id}

    def handle_command(
        self,
        session_id: str,
        plot_id: int,
        command: str,
        args: dict,
    ) -> dict:
        """Handle plot command (relayout, legendclick, selected).

        Args:
            session_id: Session identifier
            plot_id: Plot ID
            command: Command type
            args: Command arguments

        Returns:
            {"success": True, "interaction_id": int}
        """
        store, _ = get_current_session()
        if not store:
            return {"success": False, "error": "No session"}

        # Load plot JSON
        plot_store = get_plot_store()
        fig_json = plot_store.get_plot(session_id, plot_id)
        if not fig_json:
            return {"success": False, "error": f"Plot {plot_id} not found"}

        # Apply command
        if command == "relayout":
            fig_json, payload = self._apply_relayout(fig_json, args)
        elif command == "legendclick":
            fig_json, payload = self._apply_legendclick(fig_json, args)
        elif command == "selected":
            fig_json, payload = self._apply_selected(fig_json, args)
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

        # Save updated JSON
        plot_store.update_plot(session_id, plot_id, fig_json)

        # Log interaction
        interaction_id = store.log_interaction(
            session_id=session_id,
            plot_id=plot_id,
            event_type=command,
            payload=payload,
        )

        # Generate screenshot
        screenshot_path, screenshot_size = self._generate_screenshot(
            session_id, plot_id, interaction_id, fig_json
        )

        # Update interaction with screenshot
        if screenshot_path:
            store.update_interaction_screenshot(
                session_id, interaction_id, screenshot_path, screenshot_size
            )

        return {"success": True, "interaction_id": interaction_id}

    def _apply_relayout(
        self, fig_json: dict, args: dict
    ) -> tuple[dict, dict]:
        """Apply relayout command to figure.

        Args:
            fig_json: Figure JSON
            args: Relayout arguments (xaxis.range[0], etc.)

        Returns:
            (updated_fig_json, payload)
        """
        layout = fig_json.get("layout", {})
        payload = {}

        # Handle x-axis range
        if "xaxis.range[0]" in args or "xaxis.range[1]" in args:
            if "xaxis" not in layout:
                layout["xaxis"] = {}
            x_range = layout["xaxis"].get("range", [None, None])
            if "xaxis.range[0]" in args:
                x_range[0] = args["xaxis.range[0]"]
                payload["xaxis.range[0]"] = args["xaxis.range[0]"]
            if "xaxis.range[1]" in args:
                x_range[1] = args["xaxis.range[1]"]
                payload["xaxis.range[1]"] = args["xaxis.range[1]"]
            layout["xaxis"]["range"] = x_range

        # Handle y-axis range
        if "yaxis.range[0]" in args or "yaxis.range[1]" in args:
            if "yaxis" not in layout:
                layout["yaxis"] = {}
            y_range = layout["yaxis"].get("range", [None, None])
            if "yaxis.range[0]" in args:
                y_range[0] = args["yaxis.range[0]"]
                payload["yaxis.range[0]"] = args["yaxis.range[0]"]
            if "yaxis.range[1]" in args:
                y_range[1] = args["yaxis.range[1]"]
                payload["yaxis.range[1]"] = args["yaxis.range[1]"]
            layout["yaxis"]["range"] = y_range

        fig_json["layout"] = layout
        return fig_json, payload

    def _apply_legendclick(
        self, fig_json: dict, args: dict
    ) -> tuple[dict, dict]:
        """Apply legendclick command to toggle trace visibility.

        Args:
            fig_json: Figure JSON
            args: {"curve_number": int}

        Returns:
            (updated_fig_json, payload)
        """
        curve_number = args.get("curve_number", 0)
        data = fig_json.get("data", [])

        if 0 <= curve_number < len(data):
            trace = data[curve_number]
            current_visible = trace.get("visible", True)

            # Toggle visibility
            if current_visible == "legendonly":
                trace["visible"] = True
                new_visible = True
            else:
                trace["visible"] = "legendonly"
                new_visible = False

            data[curve_number] = trace
            fig_json["data"] = data

            payload = {"curve_number": curve_number, "visible": new_visible}
        else:
            payload = {"curve_number": curve_number, "error": "Invalid curve number"}

        return fig_json, payload

    def _apply_selected(
        self, fig_json: dict, args: dict
    ) -> tuple[dict, dict]:
        """Apply selection command (treated as zoom region).

        Args:
            fig_json: Figure JSON
            args: {"x_range": [min, max], "y_range": [min, max]}

        Returns:
            (updated_fig_json, payload)
        """
        payload = {}

        # Selection can act as zoom
        x_range = args.get("x_range")
        y_range = args.get("y_range")
        point_indices = args.get("point_indices")

        if x_range:
            payload["x_range"] = x_range
        if y_range:
            payload["y_range"] = y_range
        if point_indices:
            payload["point_indices"] = point_indices

        # Apply as relayout if ranges provided
        if x_range or y_range:
            relayout_args = {}
            if x_range:
                relayout_args["xaxis.range[0]"] = x_range[0]
                relayout_args["xaxis.range[1]"] = x_range[1]
            if y_range:
                relayout_args["yaxis.range[0]"] = y_range[0]
                relayout_args["yaxis.range[1]"] = y_range[1]
            fig_json, _ = self._apply_relayout(fig_json, relayout_args)

        return fig_json, payload

    def _generate_screenshot(
        self,
        session_id: str,
        plot_id: int,
        interaction_id: int,
        fig_json: dict,
    ) -> tuple[str | None, int]:
        """Generate screenshot using Kaleido.

        Args:
            session_id: Session identifier
            plot_id: Plot ID
            interaction_id: Interaction ID
            fig_json: Figure JSON

        Returns:
            (relative_path, size_kb) or (None, 0)
        """
        try:
            store, _ = get_current_session()
            if not store:
                return None, 0

            # Get screenshot directory (matches original: screenshots/{interaction_id}.png)
            screenshot_dir = store.get_screenshot_dir(session_id)
            screenshot_path = screenshot_dir / f"{interaction_id}.png"

            # Create figure and save
            fig = go.Figure(fig_json)
            fig.write_image(
                str(screenshot_path),
                format="png",
                width=self.screenshot_width,
                height=self.screenshot_height,
                scale=self.screenshot_scale,
            )

            # Get file size
            size_kb = int(screenshot_path.stat().st_size / 1024)

            # Return relative path (matches original format)
            relative_path = f"screenshots/{interaction_id}.png"
            return relative_path, size_kb

        except Exception as e:
            print(f"Screenshot generation failed: {e}")
            return None, 0


# Global singleton
_headless_handler: HeadlessHandler | None = None


def get_headless_handler() -> HeadlessHandler:
    """Get global headless handler instance."""
    global _headless_handler
    if _headless_handler is None:
        _headless_handler = HeadlessHandler()
    return _headless_handler
