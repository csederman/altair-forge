"""Heatmap plotter for Altair Forge."""

from __future__ import annotations

import altair as alt
import pandas as pd
import numpy as np
import typing as t

from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted

from ..utils import get_domain


def extract_linkage_matrix(agg: AgglomerativeClustering) -> np.ndarray:
    """Returns a linkage matrix from agglomerative clustering._summary_

    Parameters
    ----------
    agg : AgglomerativeClustering
        Agglomerative clustering object.

    Returns
    -------
    np.ndarray
        The extracted linkage matrix.
    """
    check_is_fitted(agg)

    counts = np.zeros(agg.children_.shape[0])
    n_samples = len(agg.labels_)
    for i, merge in enumerate(agg.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    cols = [agg.children_, agg.distances_, counts]

    return np.column_stack(cols).astype(float)


def get_df_coord(den: t.Dict[str, t.Any]) -> pd.DataFrame:
    """Get dendrogram coordinates as a `pd.DataFrame`.

    Parameters
    ----------
    den : t.Dict[str, t.Any]
        Dendrogram data obtained using `scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the dendrogram coordinates.
    """
    cols_xk = ["xk1", "xk2", "xk3", "xk4"]
    cols_yk = ["yk1", "yk2", "yk3", "yk4"]

    df_coord = pd.merge(
        pd.DataFrame(den["icoord"], columns=cols_xk),
        pd.DataFrame(den["dcoord"], columns=cols_yk),
        left_index=True,
        right_index=True,
    )
    return df_coord


class ClusterHeatmapBuilder:

    def __init__(
        self,
        data: pd.DataFrame,
        na_frac: float = 0.2,
        height: int = 500,
        width: int = 900,
        zero_center: bool = True,
        legend_title: str | None = None,
        legend_config: alt.LegendConfig | None = None,
        row_dendro_size: float = 40,
        row_margin_data: pd.DataFrame | None = None,
        row_margin_x_scale: alt.Scale | None = None,
        row_margin_y_scale: alt.Scale | None = None,
        row_margin_z_scale: alt.Scale | None = None,
        row_margin_legend_title: str | None = None,
        row_margin_legend_config: alt.LegendConfig | None = None,
        col_dendro_size: float = 40,
        col_margin_data: pd.DataFrame | None = None,
        col_margin_x_scale: alt.Scale | None = None,
        col_margin_y_scale: alt.Scale | None = None,
        col_margin_z_scale: alt.Scale | None = None,
        col_margin_legend_title: str | None = None,
        col_margin_legend_config: alt.LegendConfig | None = None,
    ) -> None:
        self.data = data
        self.na_frac = na_frac
        self.height = height
        self.width = width
        self.zero_center = zero_center

        if legend_config is None:
            legend_config = alt.LegendConfig()

        self.legend_config = legend_config
        self.legend_title = legend_title

        self.row_dendro_size = row_dendro_size
        self.row_margin_data = row_margin_data
        self.row_margin_x_scale = (
            alt.Scale() if row_margin_x_scale is None else row_margin_x_scale
        )
        self.row_margin_y_scale = (
            alt.Scale() if row_margin_y_scale is None else row_margin_y_scale
        )
        self.row_margin_z_scale = (
            alt.Scale() if row_margin_z_scale is None else row_margin_z_scale
        )
        self.row_margin_legend_title = row_margin_legend_title
        self.row_margin_legend_config = (
            alt.LegendConfig()
            if row_margin_legend_config is None
            else row_margin_legend_config
        )

        self.col_dendro_size = col_dendro_size
        self.col_margin_data = col_margin_data
        self.col_margin_x_scale = (
            alt.Scale() if col_margin_x_scale is None else col_margin_x_scale
        )
        self.col_margin_y_scale = (
            alt.Scale() if col_margin_y_scale is None else col_margin_y_scale
        )
        self.col_margin_z_scale = (
            alt.Scale() if col_margin_z_scale is None else col_margin_z_scale
        )
        self.col_margin_legend_title = col_margin_legend_title
        self.col_margin_legend_config = (
            alt.LegendConfig()
            if col_margin_legend_config is None
            else col_margin_legend_config
        )

        self.source, self.source_i = self._impute_missing_values()

        self._get_col_dendro_data()
        self._get_row_dendro_data()

    def _impute_missing_values(self) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        """Impute missing values for clustering."""
        if not self.data.isnull().values.any():
            return self.data, self.data

        col_na_fracs = self.data.isnull().sum() / self.data.shape[0]
        keep_cols = col_na_fracs[col_na_fracs < self.na_frac].index
        source = self.data[keep_cols].copy()

        source_i = pd.DataFrame(
            KNNImputer(n_neighbors=3).fit_transform(source),
            index=source.index,
            columns=source.columns,
        )

        return source, source_i

    def _get_col_dendro_data(self) -> None:
        """Generate the dendrogram data for the columns."""
        agg_clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        _ = agg_clust.fit(self.source_i.T)

        L = extract_linkage_matrix(agg_clust)

        # self.col_dendro_data_ = dendrogram(L, no_plot=True)

        dendro_data = dendrogram(L, no_plot=True)
        dendro_df_coord = get_df_coord(dendro_data)

        x_coord_cols = ["xk1", "xk2", "xk3", "xk4"]
        x_min = dendro_df_coord[x_coord_cols].min().min()
        x_max = dendro_df_coord[x_coord_cols].max().max()
        x_scale = alt.Scale(domain=(x_min, x_max), padding=self.rect_width_ / 2)

        y_coord_cols = ["yk1", "yk2", "yk3", "yk4"]
        y_min = dendro_df_coord[y_coord_cols].min().min()
        y_max = dendro_df_coord[y_coord_cols].max().max()
        y_scale = alt.Scale(
            domain=(y_min, y_max), padding=self.rect_height_ / 2, nice=False, zero=False
        )

        self.col_dendro_data_ = dendro_data
        self.col_dendro_coord_ = dendro_df_coord
        self.col_dendro_x_scale_ = x_scale
        self.col_dendro_y_scale_ = y_scale

    def _get_row_dendro_data(self) -> None:
        """Generate the dendrogram data for the columns."""
        agg_clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        _ = agg_clust.fit(self.source_i)

        L = extract_linkage_matrix(agg_clust)

        dendro_data = dendrogram(L, no_plot=True)
        dendro_df_coord = get_df_coord(dendro_data)

        y_coord_cols = ["xk1", "xk2", "xk3", "xk4"]
        y_min = dendro_df_coord[y_coord_cols].min().min()
        y_max = dendro_df_coord[y_coord_cols].max().max()
        y_scale = alt.Scale(
            domain=(y_min, y_max), padding=self.rect_height_ / 2, nice=False, zero=False
        )

        x_coord_cols = ["yk1", "yk2", "yk3", "yk4"]
        x_min = dendro_df_coord[x_coord_cols].min().min()
        x_max = dendro_df_coord[x_coord_cols].max().max()
        x_scale = alt.Scale(domain=(x_min, x_max), padding=0, nice=False, zero=False)

        dendro_df_coord[x_coord_cols] = x_max - dendro_df_coord[x_coord_cols]

        self.row_dendro_data_ = dendro_data
        self.row_dendro_coord_ = dendro_df_coord
        self.row_dendro_x_scale_ = x_scale
        self.row_dendro_y_scale_ = y_scale

    @property
    def col_order_(self) -> t.List[t.Any]:
        return list(self.source_i.columns[self.col_dendro_data_["leaves"]])

    @property
    def n_cols_(self) -> int:
        return len(self.source_i.columns)

    @property
    def row_order_(self) -> t.List[t.Any]:
        return list(reversed(self.source_i.index[self.row_dendro_data_["leaves"]]))

    @property
    def n_rows_(self) -> int:
        return len(self.source_i.index)

    @property
    def rect_width_(self) -> float:
        return self.width / self.n_cols_

    @property
    def rect_height_(self) -> float:
        return self.height / self.n_rows_

    @property
    def _heatmap_x_axis_params(self) -> alt.Axis:
        """Generates the x-axis component."""
        axis_params = {"domainOpacity": 0, "orient": "bottom"}
        if self.col_margin_data is not None:
            axis_params.update({"labels": False, "ticks": False})
        return axis_params

    @property
    def _heatmap_y_axis_params(self) -> alt.Axis:
        """Generates the x-axis component."""
        axis_params = {"domainOpacity": 0, "orient": "right"}
        if self.row_margin_data is not None:
            axis_params.update({"labels": False, "ticks": False})
        return axis_params

    @staticmethod
    def _prepare_heatmap_data(data: pd.DataFrame) -> pd.DataFrame:
        """"""
        return (
            data.rename_axis(index="row_var", columns="col_var")
            .melt(ignore_index=False, value_name="value")
            .reset_index()
        )

    def _create_heatmap(self) -> t.Tuple[alt.Chart, alt.Chart]:
        """"""
        hm_data = self._prepare_heatmap_data(self.source)

        z_min, z_max = get_domain(hm_data["value"])
        z_mid = 0 if self.zero_center else None
        z_scale = alt.Scale(domain=(z_min, z_mid, z_max), scheme="redyellowblue")

        hm_chart = (
            alt.Chart(hm_data, view=alt.ViewConfig(stroke=None))
            .mark_rect()
            .encode(
                alt.X("col_var:O", sort=self.col_order_)
                .axis(**self._heatmap_x_axis_params)
                .title(None),
                alt.Y("row_var:O", sort=self.row_order_)
                .axis(**self._heatmap_y_axis_params)
                .title(None),
                alt.Color("value:Q", scale=z_scale).legend(None),
            )
            .properties(
                height=self.rect_height_ * self.n_rows_,
                width=self.rect_width_ * self.n_cols_,
            )
        )

        hm_legend = (
            alt.Chart(hm_data, width=1, height=1, view=alt.ViewConfig(stroke=None))
            .mark_rect(size=0)
            .encode(
                alt.Color("value:Q", scale=z_scale)
                .legend(self.legend_config)
                .title(self.legend_title)
            )
        )

        return hm_chart, hm_legend

    def _create_col_dendro(self, width: float) -> alt.LayerChart:
        """"""
        base = alt.Chart(
            self.col_dendro_coord_,
            width=width,
            height=self.col_dendro_size,
            view=alt.ViewConfig(stroke=None),
        )

        shoulder = base.mark_rule().encode(
            alt.X("xk2:Q", title=None, scale=self.col_dendro_x_scale_).axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
            alt.X2("xk3:Q"),
            alt.Y("yk2:Q", title=None).axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
        )
        arm1 = base.mark_rule().encode(
            alt.X("xk1:Q", scale=self.col_dendro_x_scale_),
            alt.Y("yk1:Q"),
            alt.Y2("yk2:Q"),
        )
        arm2 = base.mark_rule().encode(
            alt.X("xk3:Q", scale=self.col_dendro_x_scale_),
            alt.Y("yk3:Q"),
            alt.Y2("yk4:Q"),
        )

        spacer = self._create_dendro_spacer()
        dendro = alt.layer(shoulder, arm1, arm2)

        return alt.hconcat(spacer, dendro, spacing=0)

    def _create_row_dendro(self, height: float) -> alt.LayerChart:
        """"""
        base = alt.Chart(
            self.row_dendro_coord_,
            height=height,
            width=self.row_dendro_size,
            view=alt.ViewConfig(stroke=None),
        )

        shoulder = base.mark_rule().encode(
            alt.Y("xk2:Q", title=None, scale=self.row_dendro_y_scale_).axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
            alt.Y2("xk3:Q"),
            alt.X("yk2:Q", title="")
            .axis(grid=False, labels=False, ticks=False, domainOpacity=0)
            .scale(self.row_dendro_x_scale_),
        )
        arm1 = base.mark_rule().encode(
            alt.Y("xk1:Q").scale(self.row_dendro_y_scale_),
            alt.X("yk1:Q").scale(self.row_dendro_x_scale_),
            alt.X2("yk2:Q"),
        )
        arm2 = base.mark_rule().encode(
            alt.Y("xk3:Q").scale(self.row_dendro_y_scale_),
            alt.X("yk3:Q").scale(self.row_dendro_x_scale_),
            alt.X2("yk4:Q"),
        )

        return alt.layer(shoulder, arm1, arm2)

    def _create_dendro_spacer(self) -> alt.Chart:
        """"""
        y = self.row_dendro_y_scale_.domain
        x = np.floor(self.row_dendro_x_scale_.domain)

        source = pd.DataFrame({"x": x, "y": y})

        spacer = (
            alt.Chart(
                source,
                width=self.row_dendro_size,
                height=self.col_dendro_size,
                view=alt.ViewConfig(stroke=None),
            )
            .mark_point(size=0)
            .encode(
                alt.X("x:Q", axis=None, title=None, scale=self.row_dendro_x_scale_),
                alt.Y("y:Q", axis=None, title=None).scale(zero=False, nice=False),
            )
        )

        return spacer

    def _create_col_margin_map(self) -> t.Tuple[alt.Chart, alt.Chart]:
        """"""
        n_categories = self.col_margin_data["y"].nunique()

        col_margin_map = (
            alt.Chart(
                self.col_margin_data,
                width=self.rect_width_ * self.n_cols_,
                height=self.rect_height_ * n_categories,
                view=alt.ViewConfig(stroke=None),
            )
            .mark_rect(stroke="black")
            .encode(
                alt.X("x:N", sort=self.col_order_)
                .axis(domainOpacity=0)
                .scale(self.col_margin_x_scale)
                .title(None),
                alt.Y("y:N")
                .axis(domainOpacity=0, orient="right")
                .scale(self.col_margin_y_scale)
                .title(None),
                alt.Color("value:N", scale=self.col_margin_z_scale).legend(None),
            )
        )

        spacer = self._create_dendro_spacer()
        margin = alt.hconcat(spacer, col_margin_map, spacing=0)

        legend = (
            alt.Chart(
                self.col_margin_data,
                width=1,
                height=1,
                view=alt.ViewConfig(stroke=None),
            )
            .mark_rect(size=0)
            .encode(
                alt.Color("value:N")
                .scale(self.col_margin_z_scale)
                .legend(self.col_margin_legend_config)
                .title(self.col_margin_legend_title)
            )
        )

        return margin, legend

    def _create_row_margin_map(self) -> t.Tuple[alt.Chart, alt.Chart]:
        """"""
        n_categories = self.row_margin_data["x"].nunique()

        margin = (
            alt.Chart(
                self.row_margin_data,
                height=self.rect_height_ * self.n_rows_,
                width=self.rect_width_ * n_categories,
                view=alt.ViewConfig(stroke=None),
            )
            .mark_rect(stroke="black")
            .encode(
                alt.X("x:N").scale(self.row_margin_x_scale).axis(None).title(None),
                alt.Y("y:N", sort=self.row_order_)
                .scale(self.row_margin_y_scale)
                .axis(domainOpacity=0, orient="right")
                .title(None),
                alt.Color("value:N").scale(self.row_margin_z_scale).legend(None),
            )
        )

        legend = (
            alt.Chart(
                self.row_margin_data,
                width=1,
                height=1,
                view=alt.ViewConfig(stroke=None),
            )
            .mark_rect(size=0)
            .encode(
                alt.Color("value:N")
                .scale(self.row_margin_z_scale)
                .legend(self.row_margin_legend_config)
                .title(self.row_margin_legend_title)
            )
        )

        return margin, legend

    def plot(self) -> alt.ConcatChart:
        """"""
        hm, legend = self._create_heatmap()
        col_dendro = self._create_col_dendro(width=hm.width)
        row_dendro = self._create_row_dendro(height=hm.height)

        legends = [legend]

        row_1 = col_dendro
        row_2 = alt.hconcat(row_dendro, hm, spacing=0)

        if self.row_margin_data is not None:
            row_margin, row_margin_legend = self._create_row_margin_map()
            row_2 = alt.hconcat(row_2, row_margin, spacing=2)
            legends.append(row_margin_legend)

        chart = alt.vconcat(row_1, row_2, spacing=0)

        if self.col_margin_data is not None:
            col_margin, col_margin_legend = self._create_col_margin_map()
            chart = alt.vconcat(chart, col_margin, spacing=1)
            legends.insert(1, col_margin_legend)

        if len(legends) > 0:
            legend = alt.vconcat(*legends, spacing=10)

        return alt.hconcat(chart, legend)


def cluster_heatmap(
    data: pd.DataFrame,
    na_frac: float = 0.2,
    height: float = 500,
    width: float = 900,
    zero_center: bool = True,
    row_dendro_size: float = 40,
    row_margin_data: pd.DataFrame | None = None,
    row_margin_x_scale: alt.Scale | None = None,
    row_margin_y_scale: alt.Scale | None = None,
    row_margin_z_scale: alt.Scale | None = None,
    row_margin_legend_title: str | None = None,
    row_margin_legend_config: alt.LegendConfig | None = None,
    col_dendro_size: float = 40,
    col_margin_data: pd.DataFrame | None = None,
    col_margin_x_scale: alt.Scale | None = None,
    col_margin_y_scale: alt.Scale | None = None,
    col_margin_z_scale: alt.Scale | None = None,
    col_margin_legend_title: str | None = None,
    col_margin_legend_config: alt.LegendConfig | None = None,
    legend_title: str | None = None,
    legend_config: alt.LegendConfig | None = None,
) -> alt.ConcatChart:
    """"""
    builder = ClusterHeatmapBuilder(
        data=data,
        na_frac=na_frac,
        height=height,
        width=width,
        zero_center=zero_center,
        row_dendro_size=row_dendro_size,
        row_margin_data=row_margin_data,
        row_margin_x_scale=row_margin_x_scale,
        row_margin_y_scale=row_margin_y_scale,
        row_margin_z_scale=row_margin_z_scale,
        row_margin_legend_title=row_margin_legend_title,
        row_margin_legend_config=row_margin_legend_config,
        col_dendro_size=col_dendro_size,
        col_margin_data=col_margin_data,
        col_margin_x_scale=col_margin_x_scale,
        col_margin_y_scale=col_margin_y_scale,
        col_margin_z_scale=col_margin_z_scale,
        col_margin_legend_title=col_margin_legend_title,
        col_margin_legend_config=col_margin_legend_config,
        legend_title=legend_title,
        legend_config=legend_config,
    )

    return builder.plot()
