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
        row_dendro_size: float = 40,
        col_dendro_size: float = 40,
        legend_title: str | None = None,
    ) -> None:
        self.data = data
        self.na_frac = na_frac
        self.height = height
        self.width = width
        self.zero_center = zero_center
        self.row_dendro_size = row_dendro_size
        self.col_dendro_size = col_dendro_size
        self.legend_title = legend_title

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

        self.col_dendro_data_ = dendrogram(L, no_plot=True)

    def _get_row_dendro_data(self) -> None:
        """Generate the dendrogram data for the columns."""
        agg_clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        _ = agg_clust.fit(self.source_i)

        L = extract_linkage_matrix(agg_clust)

        self.row_dendro_data_ = dendrogram(L, no_plot=True)

    @property
    def col_order_(self) -> t.List[t.Any]:
        return list(self.source_i.columns[self.col_dendro_data_["leaves"]])

    @property
    def n_cols_(self) -> int:
        return len(self.col_order_)

    @property
    def row_order_(self) -> t.List[t.Any]:
        return list(reversed(self.source_i.index[self.row_dendro_data_["leaves"]]))

    @property
    def n_rows_(self) -> int:
        return len(self.row_order_)

    @property
    def rect_width_(self) -> float:
        return self.width / self.n_cols_

    @property
    def rect_height_(self) -> float:
        return self.height / self.n_rows_

    @staticmethod
    def _prepare_heatmap_data(data: pd.DataFrame) -> pd.DataFrame:
        """"""
        return (
            data.rename_axis(index="row_var", columns="col_var")
            .melt(ignore_index=False, value_name="value")
            .reset_index()
        )

    def _make_heatmap(self) -> alt.Chart:
        """"""
        hm_data = self._prepare_heatmap_data(self.source)

        z_min, z_max = get_domain(hm_data["value"])
        z_mid = 0 if self.zero_center else None
        z_scale = alt.Scale(domain=(z_min, z_mid, z_max), scheme="redyellowblue")

        hm_chart = (
            alt.Chart(hm_data, view=alt.ViewConfig(strokeOpacity=0))
            .mark_rect()
            .encode(
                alt.X("col_var:O", sort=self.col_order_)
                .axis(domainOpacity=0, labelAngle=-45)
                .title(None),
                alt.Y("row_var:O", sort=self.row_order_)
                .axis(domainOpacity=0)
                .title(None),
                alt.Color("value:Q", scale=z_scale).legend(title=self.legend_title),
            )
            .properties(
                height=self.rect_height_ * self.n_rows_,
                width=self.rect_width_ * self.n_cols_,
            )
        )

        return hm_chart

    def _make_col_dendro(self, width: float) -> alt.LayerChart:
        """"""
        df_coord = get_df_coord(self.col_dendro_data_)

        x_coord_cols = ["xk1", "xk2", "xk3", "xk4"]
        x_min = df_coord[x_coord_cols].min().min()
        x_max = df_coord[x_coord_cols].max().max()
        x_scale = alt.Scale(domain=(x_min, x_max), padding=self.rect_width_ / 2)

        base = alt.Chart(
            df_coord,
            width=width,
            height=self.col_dendro_size,
            view=alt.ViewConfig(strokeOpacity=0),
        )

        shoulder = base.mark_rule().encode(
            alt.X("xk2:Q", title=None, scale=x_scale).axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
            alt.X2("xk3:Q"),
            alt.Y("yk2:Q", title=None).axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
        )
        arm1 = base.mark_rule().encode(
            alt.X("xk1:Q", scale=x_scale), alt.Y("yk1:Q"), alt.Y2("yk2:Q")
        )
        arm2 = base.mark_rule().encode(
            alt.X("xk3:Q", scale=x_scale), alt.Y("yk3:Q"), alt.Y2("yk4:Q")
        )

        return alt.layer(shoulder, arm1, arm2)

    def _make_row_dendro(self, height: float) -> alt.LayerChart:
        """"""
        df_coord = get_df_coord(self.row_dendro_data_)

        y_coord_cols = ["xk1", "xk2", "xk3", "xk4"]
        y_min = df_coord[y_coord_cols].min().min()
        y_max = df_coord[y_coord_cols].max().max()
        y_scale = alt.Scale(domain=(y_min, y_max), padding=self.rect_height_ / 2)

        base = alt.Chart(
            df_coord,
            height=height,
            width=self.row_dendro_size,
            view=alt.ViewConfig(strokeOpacity=0),
        )

        shoulder = base.mark_rule().encode(
            alt.Y("xk2:Q", title=None, scale=y_scale).axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
            alt.Y2("xk3:Q"),
            alt.X("yk2:Q", title="").axis(
                grid=False, labels=False, ticks=False, domainOpacity=0
            ),
        )
        arm1 = base.mark_rule().encode(
            alt.Y("xk1:Q", scale=y_scale), alt.X("yk1:Q"), alt.X2("yk2:Q")
        )
        arm2 = base.mark_rule().encode(
            alt.Y("xk3:Q", scale=y_scale), alt.X("yk3:Q"), alt.X2("yk4:Q")
        )

        return alt.layer(shoulder, arm1, arm2)

    def plot(self) -> alt.ConcatChart:
        """"""
        hm = self._make_heatmap()
        dcol = self._make_col_dendro(width=hm.width)
        drow = self._make_row_dendro(height=hm.height)

        return alt.vconcat(dcol, alt.hconcat(hm, drow, spacing=0), spacing=0)


def cluster_heatmap(
    data: pd.DataFrame,
    na_frac: float = 0.2,
    height: float = 500,
    width: float = 900,
    zero_center: bool = True,
    row_dendro_size: float = 40,
    col_dendro_size: float = 40,
    legend_title: str | None = None,
) -> alt.ConcatChart:
    """"""
    builder = ClusterHeatmapBuilder(
        data=data,
        na_frac=na_frac,
        height=height,
        width=width,
        zero_center=zero_center,
        row_dendro_size=row_dendro_size,
        col_dendro_size=col_dendro_size,
        legend_title=legend_title,
    )

    return builder.plot()
