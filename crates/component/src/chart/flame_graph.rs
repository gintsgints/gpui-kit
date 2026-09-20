use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    rc::Rc,
};

use gpui::{
    AnyElement, App, BorderStyle, Bounds, Corners, ElementId, Entity, Hitbox, HitboxBehavior, Hsla,
    IntoElement, MouseButton, MouseDownEvent, MouseUpEvent, Pixels, Point, SharedString, Size,
    Window, bounds as gpui_bounds, point, px, quad, size,
};
use gpui_base::motion::{Transition, transition};
use gpui_component_macros::IntoPlot;
use smallvec::SmallVec;

use crate::{
    ActiveTheme, Colorize,
    plot::{
        Plot,
        label::{PlotLabel, TEXT_SIZE, Text, truncate_text_to_width},
        tooltip::{PlotHover, Tooltip, TooltipState},
    },
};

/// The height of one stack row.
const ROW_HEIGHT: f32 = 18.;
/// The gap left between neighbouring frames, so a stack reads as bricks.
const FRAME_GAP: f32 = 1.;
/// A frame narrower than this never paints, and neither does its subtree: every
/// descendant lies inside its parent's extent, so the whole branch is invisible.
const MIN_FRAME_WIDTH: f32 = 0.5;
/// A frame narrower than this has no room for even one glyph and an ellipsis.
const MIN_LABEL_WIDTH: f32 = 28.;
/// The space between a frame's leading edge and its label.
const LABEL_PADDING: f32 = 4.;
/// How far the cursor may travel between press and release and still count as a
/// click, rather than a drag that happened to start on a frame.
const CLICK_SLOP: Pixels = px(3.);

/// The path from a forest root to one frame, as the child index taken at each
/// level.
///
/// Identity is a path rather than an index into a flattened tree because the
/// paint walk prunes whole subtrees by pixel width; a flat index would force it
/// to visit the pruned branches just to keep the counter right.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FlamePath(SmallVec<[u32; 8]>);

impl FlamePath {
    /// The path to the `index`th root of the forest.
    pub fn root(index: usize) -> Self {
        Self(SmallVec::from_slice(&[index as u32]))
    }

    /// The path to the `index`th child of this frame.
    pub fn child(&self, index: usize) -> Self {
        let mut path = self.0.clone();
        path.push(index as u32);
        Self(path)
    }

    /// The path to this frame's parent, or `None` for a root.
    pub fn parent(&self) -> Option<Self> {
        (self.0.len() > 1).then(|| Self(self.0[..self.0.len() - 1].into()))
    }

    /// How many frames lie between this one and the root: `0` for a root.
    pub fn depth(&self) -> usize {
        self.0.len().saturating_sub(1)
    }

    /// Whether `other` is this frame or one of its ancestors.
    pub fn starts_with(&self, other: &Self) -> bool {
        self.0.starts_with(&other.0)
    }

    /// The child index taken at each level, root first.
    pub fn indices(&self) -> &[u32] {
        &self.0
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A frame's horizontal extent, in value units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Extent {
    start: f64,
    end: f64,
}

impl Extent {
    fn width(&self) -> f64 {
        (self.end - self.start).max(0.)
    }
}

/// The slice of the value axis currently on screen.
///
/// Unfocused this is the whole forest; focused it is the focused frame's
/// extent, which is why an ancestor of the focused frame paints as a full-width
/// row without any special case — its extent simply contains the domain, and
/// the pixel rect clamps to the plot.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Domain {
    start: f64,
    end: f64,
}

impl Domain {
    fn span(&self) -> f64 {
        (self.end - self.start).max(f64::MIN_POSITIVE)
    }

    /// Where `value` falls along a plot `width` pixels wide.
    fn to_px(&self, value: f64, width: f32) -> f32 {
        (((value - self.start) / self.span()) * width as f64) as f32
    }

    /// Which value sits under `x` on a plot `width` pixels wide.
    fn value_at(&self, x: f32, width: f32) -> f64 {
        self.start + (x as f64 / width.max(1.) as f64) * self.span()
    }
}

/// The hover a flame graph paints, sampled once per frame in [`Plot::hover`].
#[derive(Clone)]
struct FlameHover {
    /// The frame under the cursor, or the last one while the hover fades out.
    path: FlamePath,
    /// How far the hover has faded in.
    focus: f32,
}

/// A flame graph: a tree of stack frames, each frame as wide as its value and
/// as deep as its place in the stack.
///
/// The chart never folds, sorts or rescales the tree it is given. Aggregating
/// samples into a tree, and ordering the children within it, stay the caller's
/// decisions; a frame's own value is authoritative for its width and for every
/// percentage in the tooltip.
///
/// Zoom state is the caller's too: pass the focused frame to
/// [`FlameGraph::focus`] and update it from [`FlameGraph::on_click`]. The whole
/// state is one `Option<FlamePath>` — the ancestor rows, the zoomed domain and
/// the way back out all derive from it.
#[derive(IntoPlot)]
pub struct FlameGraph<T>
where
    T: 'static,
{
    roots: Rc<Vec<T>>,
    children: Option<Rc<dyn for<'a> Fn(&'a T) -> &'a [T]>>,
    value: Option<Rc<dyn Fn(&T) -> f64>>,
    label: Option<Rc<dyn Fn(&T) -> SharedString>>,
    color: Option<Rc<dyn Fn(&T) -> Hsla>>,
    format: Option<Rc<dyn Fn(f64) -> SharedString>>,
    focus: Option<FlamePath>,
    on_click: Option<Rc<dyn Fn(&FlamePath, &mut Window, &mut App)>>,
    row_height: Pixels,
    inverted: bool,
    id: Option<ElementId>,
    /// The plot's area, kept from `prepaint` so `hover` can resolve the cursor.
    bounds: Option<Bounds<Pixels>>,
    /// This frame's animated domain, sampled in `prepaint` so the hover, the
    /// tooltip and the paint all read the same one.
    domain: Option<Domain>,
    /// The chart's own hitbox, so a click knows whether a popup covers it.
    hitbox: Option<Hitbox>,
    hover: Option<FlameHover>,
}

impl<T> FlameGraph<T> {
    /// Build a flame graph over a forest of root frames, packed left to right.
    pub fn new<I>(roots: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::shared(Rc::new(roots.into_iter().collect()))
    }

    /// Build a flame graph over a forest the caller already holds behind an
    /// `Rc`.
    ///
    /// A profile is loaded once and kept; this adopts that tree rather than
    /// copying a hundred thousand frames on every render.
    pub fn shared(roots: Rc<Vec<T>>) -> Self {
        Self {
            roots,
            children: None,
            value: None,
            label: None,
            color: None,
            format: None,
            focus: None,
            on_click: None,
            row_height: px(ROW_HEIGHT),
            inverted: false,
            id: None,
            bounds: None,
            domain: None,
            hitbox: None,
            hover: None,
        }
    }

    /// Enable hover tooltips and click-to-zoom for this chart.
    ///
    /// The `id` must be unique among sibling elements. Without it, the chart
    /// stays a non-interactive plot.
    pub fn id(mut self, id: impl Into<ElementId>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Read a frame's children. Frames without this accessor have none.
    pub fn children(mut self, children: impl for<'a> Fn(&'a T) -> &'a [T] + 'static) -> Self {
        self.children = Some(Rc::new(children));
        self
    }

    /// Read a frame's value: its width, in whatever unit the profile counts —
    /// samples, milliseconds, bytes.
    ///
    /// A frame's value is authoritative. Children pack from their parent's
    /// start, and a child reaching past the parent's end is clipped there
    /// rather than rescaled, so a tree whose children over-run their parent
    /// shows it instead of hiding it.
    pub fn value(mut self, value: impl Fn(&T) -> f64 + 'static) -> Self {
        self.value = Some(Rc::new(value));
        self
    }

    /// Read a frame's label, drawn inside the frame when it is wide enough and
    /// shown in full in the tooltip.
    pub fn label(mut self, label: impl Fn(&T) -> SharedString + 'static) -> Self {
        self.label = Some(Rc::new(label));
        self
    }

    /// Read a frame's colour.
    ///
    /// Without this, a frame takes one of the theme's chart colours, chosen by
    /// hashing its label, so neighbours differ and a frame keeps its colour
    /// across renders.
    pub fn color(mut self, color: impl Fn(&T) -> Hsla + 'static) -> Self {
        self.color = Some(Rc::new(color));
        self
    }

    /// Format a value for the tooltip (e.g. `1.23 s`). Defaults to the bare
    /// number; percentages are always computed by the chart.
    pub fn format(mut self, format: impl Fn(f64) -> SharedString + 'static) -> Self {
        self.format = Some(Rc::new(format));
        self
    }

    /// Zoom to a frame: it fills the width, its ancestors stay as full-width
    /// rows, and everything outside its extent is clipped away.
    pub fn focus(mut self, focus: Option<FlamePath>) -> Self {
        self.focus = focus.filter(|path| !path.is_empty());
        self
    }

    /// Handle a click on a frame.
    ///
    /// Clicking a descendant zooms in, clicking an ancestor zooms back out to
    /// it, and clicking a root row resets. Requires [`FlameGraph::id`].
    pub fn on_click(
        mut self,
        on_click: impl Fn(&FlamePath, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_click = Some(Rc::new(on_click));
        self
    }

    /// Set the height of one stack row. Defaults to 18px.
    pub fn row_height(mut self, row_height: impl Into<Pixels>) -> Self {
        self.row_height = row_height.into();
        self
    }

    /// Grow upward from the bottom edge, the original flame orientation.
    ///
    /// The default is an icicle: the root on top, the stack growing down, as
    /// samply and the browser profilers draw it.
    pub fn flame(mut self) -> Self {
        self.inverted = true;
        self
    }

    /// The height a stack `depth` frames deep needs at this chart's row
    /// height, for sizing the scroll container the chart is dropped into.
    ///
    /// The chart fills the area it is given and clips what does not fit; it
    /// owns no scrolling of its own.
    pub fn height(&self, depth: usize) -> Pixels {
        self.row_height * depth as f32
    }

    /// The geometry for this frame: the tree, the animated domain and the row
    /// metrics that paint, hit-testing and the tooltip all share.
    ///
    /// `None` until `prepaint` has sampled the domain, and whenever the value
    /// accessor is missing — a tree with no value has no width to draw.
    fn geometry(&self, bounds: Bounds<Pixels>) -> Option<Geometry<T>> {
        self.geometry_with(bounds, self.domain?)
    }

    /// The same geometry against a domain the caller supplies, for `prepaint`,
    /// which has to measure the tree before it knows where the zoom has got to.
    fn geometry_with(&self, bounds: Bounds<Pixels>, domain: Domain) -> Option<Geometry<T>> {
        Some(Geometry {
            roots: self.roots.clone(),
            children: self.children.clone(),
            value: self.value.clone()?,
            domain,
            row_height: self.row_height.as_f32().max(1.),
            inverted: self.inverted,
            size: bounds.size,
        })
    }

    /// The domain to settle at: the focused frame's extent, or the whole
    /// forest when nothing is focused or the focus no longer resolves.
    fn target_domain(&self, geometry: &Geometry<T>) -> Domain {
        self.focus
            .as_ref()
            .and_then(|path| geometry.extent_of(path))
            .filter(|extent| extent.width() > 0.)
            .map(|extent| Domain {
                start: extent.start,
                end: extent.end,
            })
            .unwrap_or(Domain {
                start: 0.,
                end: geometry.total().max(f64::MIN_POSITIVE),
            })
    }

    fn label_of(&self, node: &T) -> SharedString {
        self.label
            .as_ref()
            .map(|label| label(node))
            .unwrap_or_default()
    }

    /// A frame's colour: the caller's, or one of the theme's chart colours
    /// picked by hashing the label, nudged by depth so a colour collision
    /// between a frame and the one above it still reads as two frames.
    fn color_of(&self, node: &T, depth: usize, cx: &App) -> Hsla {
        if let Some(color) = self.color.as_ref() {
            return color(node);
        }

        let theme = cx.theme();
        let palette = [
            theme.chart_1,
            theme.chart_2,
            theme.chart_3,
            theme.chart_4,
            theme.chart_5,
        ];
        let mut hasher = DefaultHasher::new();
        self.label_of(node).hash(&mut hasher);
        let base = palette[(hasher.finish() % palette.len() as u64) as usize];

        match depth % 3 {
            0 => base,
            1 => base.lighten(0.04),
            _ => base.darken(0.04),
        }
    }

    fn format_value(&self, value: f64) -> SharedString {
        match self.format.as_ref() {
            Some(format) => format(value),
            None => SharedString::from(format!("{}", value)),
        }
    }
}

/// The tree plus the metrics that map it to pixels.
///
/// Held apart from the chart so the mouse handler — which outlives the paint
/// and cannot borrow the chart — can own a copy of everything hit-testing
/// needs.
struct Geometry<T> {
    roots: Rc<Vec<T>>,
    children: Option<Rc<dyn for<'a> Fn(&'a T) -> &'a [T]>>,
    value: Rc<dyn Fn(&T) -> f64>,
    domain: Domain,
    row_height: f32,
    inverted: bool,
    size: Size<Pixels>,
}

impl<T> Clone for Geometry<T> {
    fn clone(&self) -> Self {
        Self {
            roots: self.roots.clone(),
            children: self.children.clone(),
            value: self.value.clone(),
            domain: self.domain,
            row_height: self.row_height,
            inverted: self.inverted,
            size: self.size,
        }
    }
}

impl<T> Geometry<T> {
    /// A frame's value, with a negative, infinite or NaN one read as zero so a
    /// malformed profile prunes instead of poisoning the layout.
    fn value_of(&self, node: &T) -> f64 {
        let value = (self.value)(node);
        if value.is_finite() && value > 0. {
            value
        } else {
            0.
        }
    }

    fn children_of<'a>(&self, node: &'a T) -> &'a [T] {
        self.children
            .as_ref()
            .map(|children| children(node))
            .unwrap_or(&[])
    }

    fn total(&self) -> f64 {
        self.roots.iter().map(|node| self.value_of(node)).sum()
    }

    fn width(&self) -> f32 {
        self.size.width.as_f32()
    }

    /// Walk down `path`, accumulating the offsets of the siblings passed at
    /// each level. Returns `None` for a path that no longer resolves, so a
    /// stale focus falls back to the full view instead of panicking.
    fn extent_of(&self, path: &FlamePath) -> Option<Extent> {
        let mut nodes: &[T] = &self.roots;
        let mut extent = Extent {
            start: 0.,
            end: self.total(),
        };

        for index in path.indices() {
            let index = *index as usize;
            let mut start = extent.start;
            for sibling in nodes.iter().take(index) {
                start += self.value_of(sibling).min((extent.end - start).max(0.));
            }

            let node = nodes.get(index)?;
            if start >= extent.end {
                return None;
            }
            let width = self.value_of(node).min(extent.end - start);
            if width <= 0. {
                return None;
            }

            extent = Extent {
                start,
                end: start + width,
            };
            nodes = self.children_of(node);
        }

        Some(extent)
    }

    fn node_at(&self, path: &FlamePath) -> Option<&T> {
        let mut nodes: &[T] = &self.roots;
        let mut found = None;
        for index in path.indices() {
            let node = nodes.get(*index as usize)?;
            nodes = self.children_of(node);
            found = Some(node);
        }
        found
    }

    /// The pixel rect of a frame at `depth`, clamped to the plot.
    ///
    /// Clamping is what makes an ancestor of the focused frame paint as a
    /// full-width row: its extent runs past both edges of the domain.
    fn rect_of(&self, extent: Extent, depth: usize) -> Bounds<Pixels> {
        let width = self.width();
        let x0 = self.domain.to_px(extent.start, width).max(0.);
        let x1 = self.domain.to_px(extent.end, width).min(width);
        let top = match self.inverted {
            true => self.size.height.as_f32() - (depth + 1) as f32 * self.row_height,
            false => depth as f32 * self.row_height,
        };

        gpui_bounds(
            point(px(x0), px(top)),
            size(
                px((x1 - x0 - FRAME_GAP).max(0.)),
                px(self.row_height - FRAME_GAP),
            ),
        )
    }

    /// Visit every frame worth painting, deepest branch last.
    ///
    /// A frame narrower than [`MIN_FRAME_WIDTH`] stops the walk there: its
    /// descendants all sit inside its extent, so none of them can be visible
    /// either. This is what keeps a 100k-frame profile's cost proportional to
    /// the pixels on screen rather than to the tree.
    fn walk(&self, mut visit: impl FnMut(&T, usize, Extent, Bounds<Pixels>)) {
        let width = self.width();
        let height = self.size.height.as_f32();
        let mut stack: Vec<(&[T], usize, Extent)> = vec![(
            &self.roots,
            0,
            Extent {
                start: 0.,
                end: self.total(),
            },
        )];

        while let Some((nodes, depth, parent)) = stack.pop() {
            // Rows past the far edge of the plot, and everything below them.
            let row_top = match self.inverted {
                true => height - (depth + 1) as f32 * self.row_height,
                false => depth as f32 * self.row_height,
            };
            if self.inverted && row_top + self.row_height <= 0. {
                continue;
            }
            if !self.inverted && row_top >= height {
                continue;
            }

            let mut start = parent.start;
            for node in nodes {
                if start >= parent.end {
                    break;
                }
                let value = self.value_of(node).min(parent.end - start);
                let extent = Extent {
                    start,
                    end: start + value,
                };
                start = extent.end;
                if value <= 0. {
                    continue;
                }

                let x0 = self.domain.to_px(extent.start, width).max(0.);
                let x1 = self.domain.to_px(extent.end, width).min(width);
                if x1 - x0 < MIN_FRAME_WIDTH {
                    continue;
                }

                visit(node, depth, extent, self.rect_of(extent, depth));
                let children = self.children_of(node);
                if !children.is_empty() {
                    stack.push((children, depth + 1, extent));
                }
            }
        }
    }

    /// The frame under a point in plot-local coordinates.
    fn path_at(&self, position: Point<Pixels>) -> Option<FlamePath> {
        let width = self.width();
        let height = self.size.height.as_f32();
        let (x, y) = (position.x.as_f32(), position.y.as_f32());
        if x < 0. || x > width || y < 0. || y > height {
            return None;
        }

        let row = match self.inverted {
            true => ((height - y) / self.row_height).floor(),
            false => (y / self.row_height).floor(),
        };
        if row < 0. {
            return None;
        }
        let row = row as usize;

        let value = self.domain.value_at(x, width);
        let mut nodes: &[T] = &self.roots;
        let mut path = FlamePath::default();
        let mut parent = Extent {
            start: 0.,
            end: self.total(),
        };

        for _ in 0..=row {
            let mut start = parent.start;
            let mut found = None;
            for (index, node) in nodes.iter().enumerate() {
                if start >= parent.end {
                    break;
                }
                let end = start + self.value_of(node).min(parent.end - start);
                if value >= start && value < end {
                    found = Some((index, node, Extent { start, end }));
                    break;
                }
                start = end;
            }

            // A gap at this depth: the parent's own time, which is not a frame.
            let (index, node, extent) = found?;
            path = path.child(index);
            parent = extent;
            nodes = self.children_of(node);
        }

        Some(path)
    }
}

/// The press a click is measured against, kept in element state between the
/// mouse-down and the mouse-up.
#[derive(Default)]
struct PressState(Option<Point<Pixels>>);

impl<T> Plot for FlameGraph<T> {
    fn prepaint(
        &mut self,
        bounds: Bounds<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Vec<AnyElement> {
        self.bounds = Some(bounds);

        // The domain is sampled once here, before `hover`, `tooltip` and
        // `paint` run, so all three agree on where the zoom has got to and a
        // click mid-flight hits what is actually under the cursor.
        let full = Domain {
            start: 0.,
            end: f64::MIN_POSITIVE,
        };
        if let Some(geometry) = self.geometry_with(bounds, full) {
            let total = geometry.total().max(f64::MIN_POSITIVE);
            let target = self.target_domain(&geometry);

            self.domain = Some(match self.id.clone() {
                // Animate in normalised domain units: a profile counted in
                // nanoseconds would quantise visibly if f32 carried the raw
                // values, while 0..1 endpoints keep their precision.
                Some(id) => {
                    let policy = Transition::new(cx.theme().motion_tokens().duration_slow);
                    let start = transition(
                        ElementId::NamedChild(id.clone().into(), "flame-domain-start".into()),
                        (target.start / total) as f32,
                        policy.clone(),
                        window,
                        cx,
                    );
                    let end = transition(
                        ElementId::NamedChild(id.into(), "flame-domain-end".into()),
                        (target.end / total) as f32,
                        policy,
                        window,
                        cx,
                    );
                    Domain {
                        start: start as f64 * total,
                        end: end as f64 * total,
                    }
                }
                None => target,
            });
        }

        if self.id.is_some() {
            self.hitbox = Some(window.insert_hitbox(bounds, HitboxBehavior::Normal));
        }

        vec![]
    }

    fn paint(&mut self, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
        let Some(geometry) = self.geometry(bounds) else {
            return;
        };

        let radius = cx.theme().radius.as_f32();
        let origin = bounds.origin;
        let mut labels: Vec<Text> = vec![];

        geometry.walk(|node, depth, _, rect| {
            let color = self.color_of(node, depth, cx);
            let width = rect.size.width.as_f32();
            let radii = Corners::all(px(radius
                .min(width / 2.)
                .min(rect.size.height.as_f32() / 2.)));
            window.paint_quad(
                gpui::fill(gpui_bounds(rect.origin + origin, rect.size), color).corner_radii(radii),
            );

            if width < MIN_LABEL_WIDTH {
                return;
            }
            let label = self.label_of(node);
            if label.is_empty() {
                return;
            }
            let text =
                truncate_text_to_width(&label, px(TEXT_SIZE), width - LABEL_PADDING * 2., window);
            labels.push(Text::new(
                text,
                point(
                    rect.origin.x + px(LABEL_PADDING),
                    rect.origin.y + px((geometry.row_height - FRAME_GAP - TEXT_SIZE) / 2.),
                ),
                label_color(color, cx),
            ));
        });

        // The hovered frame, painted over the walk and under the labels so its
        // own label stays legible.
        if let Some(hover) = self.hover.clone() {
            if let (Some(node), Some(extent)) = (
                geometry.node_at(&hover.path),
                geometry.extent_of(&hover.path),
            ) {
                let rect = geometry.rect_of(extent, hover.path.depth());
                let color = self.color_of(node, hover.path.depth(), cx);
                let radii = Corners::all(px(radius
                    .min(rect.size.width.as_f32() / 2.)
                    .min(rect.size.height.as_f32() / 2.)));
                window.paint_quad(quad(
                    gpui_bounds(rect.origin + origin, rect.size),
                    radii,
                    color.lighten(0.08 * hover.focus),
                    px(1.),
                    cx.theme().foreground.opacity(0.4 * hover.focus),
                    BorderStyle::default(),
                ));
            }
        }

        PlotLabel::new(labels).paint(&bounds, window, cx);

        self.register_click(bounds, geometry, window, cx);
    }

    fn id(&self) -> Option<ElementId> {
        self.id.clone()
    }

    fn tooltip_state(
        &self,
        position: Point<Pixels>,
        bounds: Bounds<Pixels>,
        _cx: &App,
    ) -> Option<TooltipState> {
        // The hovered frame re-derives from the cursor in `hover` and
        // `tooltip`: walking down to it costs one step per level, so there is
        // nothing worth carrying in the index or the dots.
        // `position` already arrives relative to the plot's origin.
        let geometry = self.geometry(bounds)?;
        geometry.path_at(position)?;
        Some(TooltipState::new(0, position, vec![]))
    }

    fn hover(&mut self, hover: Option<&PlotHover>, _window: &mut Window, _cx: &mut App) {
        let Some(bounds) = self.bounds else {
            self.hover = None;
            return;
        };
        let Some(geometry) = self.geometry(bounds) else {
            self.hover = None;
            return;
        };

        self.hover = hover.and_then(|hover| {
            Some(FlameHover {
                path: geometry.path_at(hover.state().cross_line)?,
                focus: hover.focus(),
            })
        });
    }

    fn tooltip(
        &self,
        state: &TooltipState,
        cursor: Point<Pixels>,
        bounds: Bounds<Pixels>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Option<AnyElement> {
        let geometry = self.geometry(bounds)?;
        let path = match self.hover.as_ref() {
            Some(hover) => hover.path.clone(),
            None => geometry.path_at(state.cross_line)?,
        };
        let node = geometry.node_at(&path)?;

        let value = geometry.value_of(node);
        let children: f64 = geometry
            .children_of(node)
            .iter()
            .map(|child| geometry.value_of(child))
            .sum();
        let total = geometry.total();
        let parent = path
            .parent()
            .and_then(|parent| {
                geometry
                    .node_at(&parent)
                    .map(|node| geometry.value_of(node))
            })
            .unwrap_or(total);
        let color = self.color_of(node, path.depth(), cx);
        let percent = |part: f64, whole: f64| {
            SharedString::from(match whole > 0. {
                true => format!("{:.2}%", part / whole * 100.),
                false => "—".to_string(),
            })
        };

        Some(
            Tooltip::new(cursor, bounds.size)
                .gap(px(8.))
                .title(self.label_of(node))
                .row(color, "Total", self.format_value(value))
                .row(color, "Self", self.format_value((value - children).max(0.)))
                .row(color, "% of all", percent(value, total))
                .row(color, "% of parent", percent(value, parent))
                .into_any_element(),
        )
    }
}

impl<T> FlameGraph<T> {
    /// Route clicks to [`FlameGraph::on_click`].
    ///
    /// The handler outlives the paint, so it owns `Rc` clones of the tree and
    /// the geometry rather than borrowing the chart. A press that travels more
    /// than [`CLICK_SLOP`] is a drag — scrolling a deep stack must not end in
    /// an accidental zoom — and a press under an open popup is not ours.
    fn register_click(
        &self,
        bounds: Bounds<Pixels>,
        geometry: Geometry<T>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let (Some(id), Some(hitbox), Some(on_click)) =
            (self.id.clone(), self.hitbox.clone(), self.on_click.clone())
        else {
            return;
        };

        let press: Entity<PressState> = window.use_keyed_state(
            ElementId::NamedChild(id.into(), "flame-press".into()),
            cx,
            |_, _| PressState::default(),
        );

        window.on_mouse_event({
            let press = press.clone();
            let hitbox = hitbox.clone();
            move |event: &MouseDownEvent, phase, window: &mut Window, cx: &mut App| {
                if !phase.bubble() || event.button != MouseButton::Left {
                    return;
                }
                let inside = hitbox.is_hovered(window);
                press.update(cx, |press, _| {
                    press.0 = inside.then_some(event.position);
                });
            }
        });

        window.on_mouse_event(
            move |event: &MouseUpEvent, phase, window: &mut Window, cx: &mut App| {
                if !phase.bubble() || event.button != MouseButton::Left {
                    return;
                }
                let Some(start) = press.update(cx, |press, _| press.0.take()) else {
                    return;
                };
                let travel = event.position - start;
                if travel.x.abs() > CLICK_SLOP || travel.y.abs() > CLICK_SLOP {
                    return;
                }
                if !hitbox.is_hovered(window) {
                    return;
                }
                let Some(path) = geometry.path_at(event.position - bounds.origin) else {
                    return;
                };
                on_click(&path, window, cx);
            },
        );
    }
}

/// Whichever of the theme's foreground and background reads against `fill`.
///
/// A frame's colour comes from the caller or from a hash, so neither end of the
/// theme is reliably the readable one.
fn label_color(fill: Hsla, cx: &App) -> Hsla {
    let (foreground, background) = (cx.theme().foreground, cx.theme().background);
    match (fill.l - foreground.l).abs() >= (fill.l - background.l).abs() {
        true => foreground,
        false => background,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Node {
        label: SharedString,
        value: f64,
        children: Vec<Node>,
    }

    impl Node {
        fn new(label: &str, value: f64, children: Vec<Node>) -> Self {
            Self {
                label: label.into(),
                value,
                children,
            }
        }
    }

    /// A geometry over `roots`, 100px wide and 5 rows tall, showing everything.
    fn geometry(roots: Vec<Node>) -> Geometry<Node> {
        let total = roots.iter().map(|node| node.value).sum();
        Geometry {
            roots: Rc::new(roots),
            children: Some(Rc::new(|node: &Node| node.children.as_slice())),
            value: Rc::new(|node: &Node| node.value),
            domain: Domain {
                start: 0.,
                end: total,
            },
            row_height: 20.,
            inverted: false,
            size: size(px(100.), px(100.)),
        }
    }

    fn tree() -> Vec<Node> {
        vec![
            Node::new(
                "main",
                60.,
                vec![Node::new("parse", 20., vec![Node::new("lex", 5., vec![])])],
            ),
            Node::new("worker", 40., vec![]),
        ]
    }

    #[test]
    fn test_flame_graph_builder() {
        let chart = FlameGraph::new(tree())
            .id("flame")
            .children(|node: &Node| node.children.as_slice())
            .value(|node: &Node| node.value)
            .label(|node: &Node| node.label.clone())
            .color(|_| gpui::red())
            .format(|value| format!("{value} ms").into())
            .focus(Some(FlamePath::root(0).child(0)))
            .on_click(|_, _, _| {})
            .row_height(px(24.))
            .flame();

        assert_eq!(chart.roots.len(), 2);
        assert!(chart.id.is_some());
        assert!(chart.children.is_some());
        assert!(chart.value.is_some());
        assert!(chart.label.is_some());
        assert!(chart.color.is_some());
        assert!(chart.on_click.is_some());
        assert_eq!(chart.focus, Some(FlamePath::root(0).child(0)));
        assert_eq!(chart.row_height, px(24.));
        assert!(chart.inverted);
        assert_eq!(chart.format_value(12.), SharedString::from("12 ms"));
    }

    #[test]
    fn an_empty_focus_is_no_focus() {
        // A root path addresses a root frame; the empty path addresses nothing,
        // so it must not zoom the chart to a zero-width domain.
        let chart = FlameGraph::new(tree()).focus(Some(FlamePath::default()));
        assert_eq!(chart.focus, None);
    }

    #[test]
    fn siblings_pack_from_their_parent_start() {
        let geometry = geometry(tree());

        assert_eq!(
            geometry.extent_of(&FlamePath::root(0)),
            Some(Extent {
                start: 0.,
                end: 60.
            })
        );
        // The second root starts where the first ends.
        assert_eq!(
            geometry.extent_of(&FlamePath::root(1)),
            Some(Extent {
                start: 60.,
                end: 100.
            })
        );
        // A child packs from its parent's start, not from zero.
        assert_eq!(
            geometry.extent_of(&FlamePath::root(0).child(0)),
            Some(Extent {
                start: 0.,
                end: 20.
            })
        );
    }

    #[test]
    fn children_overrunning_a_parent_are_clipped_not_rescaled() {
        // 8 + 8 children under a parent of 10: the tree is inconsistent, and
        // the parent's own value stays authoritative.
        let geometry = geometry(vec![Node::new(
            "parent",
            10.,
            vec![Node::new("a", 8., vec![]), Node::new("b", 8., vec![])],
        )]);

        assert_eq!(
            geometry.extent_of(&FlamePath::root(0).child(0)),
            Some(Extent { start: 0., end: 8. })
        );
        // Clipped at the parent's end rather than scaled to fit beside it.
        assert_eq!(
            geometry.extent_of(&FlamePath::root(0).child(1)),
            Some(Extent {
                start: 8.,
                end: 10.
            })
        );
        assert_eq!(
            geometry.extent_of(&FlamePath::root(0)).unwrap().width(),
            10.
        );
    }

    #[test]
    fn a_child_with_no_room_left_does_not_resolve() {
        let geometry = geometry(vec![Node::new(
            "parent",
            10.,
            vec![Node::new("a", 10., vec![]), Node::new("b", 5., vec![])],
        )]);

        assert_eq!(geometry.extent_of(&FlamePath::root(0).child(1)), None);
    }

    #[test]
    fn a_stale_path_does_not_resolve() {
        let geometry = geometry(tree());

        assert_eq!(geometry.extent_of(&FlamePath::root(7)), None);
        assert_eq!(geometry.extent_of(&FlamePath::root(1).child(0)), None);
    }

    #[test]
    fn a_non_positive_value_prunes_the_frame() {
        let geometry = geometry(vec![
            Node::new("zero", 0., vec![Node::new("child", 5., vec![])]),
            Node::new("nan", f64::NAN, vec![]),
            Node::new("negative", -5., vec![]),
            Node::new("real", 10., vec![]),
        ]);

        assert_eq!(geometry.total(), 10.);
        assert_eq!(geometry.extent_of(&FlamePath::root(0)), None);
        assert_eq!(geometry.extent_of(&FlamePath::root(1)), None);
        assert_eq!(geometry.extent_of(&FlamePath::root(2)), None);
        assert_eq!(
            geometry.extent_of(&FlamePath::root(3)),
            Some(Extent {
                start: 0.,
                end: 10.
            })
        );
    }

    #[test]
    fn a_zoomed_domain_spans_the_focused_frame() {
        let chart = FlameGraph::new(tree())
            .children(|node: &Node| node.children.as_slice())
            .value(|node: &Node| node.value)
            .focus(Some(FlamePath::root(0).child(0)));
        let geometry = geometry(tree());

        let domain = chart.target_domain(&geometry);
        assert_eq!(domain.start, 0.);
        assert_eq!(domain.end, 20.);

        // The focused frame fills the width, and its ancestor runs past both
        // edges — which is what makes an ancestor paint as a full-width row.
        assert_eq!(domain.to_px(0., 100.), 0.);
        assert_eq!(domain.to_px(20., 100.), 100.);
        assert_eq!(domain.to_px(60., 100.), 300.);
    }

    #[test]
    fn an_unresolvable_focus_falls_back_to_the_whole_forest() {
        let chart = FlameGraph::new(tree())
            .children(|node: &Node| node.children.as_slice())
            .value(|node: &Node| node.value)
            .focus(Some(FlamePath::root(9)));

        let domain = chart.target_domain(&geometry(tree()));
        assert_eq!(domain.start, 0.);
        assert_eq!(domain.end, 100.);
    }

    #[test]
    fn the_walk_prunes_subpixel_frames_and_their_subtrees() {
        // "thin" is 0.4% of the forest: under half a pixel at 100px wide, so
        // neither it nor its (proportionally wider) child may be visited.
        let geometry = geometry(vec![
            Node::new("wide", 99.6, vec![]),
            Node::new("thin", 0.4, vec![Node::new("under-thin", 0.4, vec![])]),
        ]);

        let mut visited = vec![];
        geometry.walk(|node, depth, _, _| visited.push((node.label.clone(), depth)));

        assert_eq!(visited, vec![(SharedString::from("wide"), 0)]);
    }

    #[test]
    fn the_walk_reports_a_frame_rect_inset_by_the_gap() {
        let geometry = geometry(vec![Node::new("only", 100., vec![])]);

        let mut rects = vec![];
        geometry.walk(|_, _, _, rect| rects.push(rect));

        let rect = rects[0];
        assert_eq!(rect.origin, point(px(0.), px(0.)));
        assert_eq!(rect.size.width, px(100. - FRAME_GAP));
        assert_eq!(rect.size.height, px(20. - FRAME_GAP));
    }

    #[test]
    fn a_point_resolves_to_the_frame_under_it() {
        let geometry = geometry(tree());

        // Row 0, left half: the first root.
        assert_eq!(
            geometry.path_at(point(px(30.), px(10.))),
            Some(FlamePath::root(0))
        );
        // Row 0, past 60%: the second root.
        assert_eq!(
            geometry.path_at(point(px(80.), px(10.))),
            Some(FlamePath::root(1))
        );
        // Row 1 over the first root's child.
        assert_eq!(
            geometry.path_at(point(px(10.), px(25.))),
            Some(FlamePath::root(0).child(0))
        );
        // Row 2 over the grandchild.
        assert_eq!(
            geometry.path_at(point(px(2.), px(45.))),
            Some(FlamePath::root(0).child(0).child(0))
        );
    }

    #[test]
    fn a_point_over_self_time_hits_nothing() {
        let geometry = geometry(tree());

        // Row 1 at x = 40: inside "main", but past its only child, so the
        // cursor is over main's own time, which is not a frame.
        assert_eq!(geometry.path_at(point(px(40.), px(25.))), None);
        // Below the deepest frame on this branch.
        assert_eq!(geometry.path_at(point(px(2.), px(65.))), None);
    }

    #[test]
    fn a_point_outside_the_plot_hits_nothing() {
        let geometry = geometry(tree());

        assert_eq!(geometry.path_at(point(px(-1.), px(10.))), None);
        assert_eq!(geometry.path_at(point(px(30.), px(-1.))), None);
        assert_eq!(geometry.path_at(point(px(101.), px(10.))), None);
    }

    #[test]
    fn an_inverted_graph_counts_rows_from_the_bottom() {
        let mut geometry = geometry(tree());
        geometry.inverted = true;

        // The root row sits against the bottom edge.
        assert_eq!(
            geometry.path_at(point(px(30.), px(95.))),
            Some(FlamePath::root(0))
        );
        assert_eq!(
            geometry.path_at(point(px(10.), px(75.))),
            Some(FlamePath::root(0).child(0))
        );
        assert_eq!(
            geometry
                .rect_of(
                    Extent {
                        start: 0.,
                        end: 60.
                    },
                    0
                )
                .origin,
            point(px(0.), px(80.))
        );
    }

    #[test]
    fn a_path_addresses_its_ancestors() {
        let path = FlamePath::root(1).child(2).child(3);

        assert_eq!(path.depth(), 2);
        assert_eq!(path.indices(), &[1, 2, 3]);
        assert_eq!(path.parent(), Some(FlamePath::root(1).child(2)));
        assert_eq!(FlamePath::root(1).parent(), None);
        assert!(path.starts_with(&FlamePath::root(1)));
        assert!(!path.starts_with(&FlamePath::root(0)));
    }
}

#[cfg(test)]
mod interaction_tests {
    use std::cell::RefCell;

    use gpui::{
        Context, IntoElement, Modifiers, MouseButton, ParentElement as _, Render, Styled as _,
        TestAppContext, Window, div, point, px,
    };

    use super::*;

    /// How far the chart sits from the window's top-left in these tests.
    const INSET: Pixels = px(40.);

    struct Node {
        label: SharedString,
        value: f64,
        children: Vec<Node>,
    }

    /// Two roots, 60/40, the first with one child covering its left third.
    fn tree() -> Vec<Node> {
        vec![
            Node {
                label: "main".into(),
                value: 60.,
                children: vec![Node {
                    label: "parse".into(),
                    value: 20.,
                    children: vec![],
                }],
            },
            Node {
                label: "worker".into(),
                value: 40.,
                children: vec![],
            },
        ]
    }

    /// A window-filling flame graph that records every frame clicked.
    struct FlameView {
        clicked: Rc<RefCell<Vec<FlamePath>>>,
    }

    impl Render for FlameView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let clicked = self.clicked.clone();
            // Inset the chart so its bounds origin is not the window's: a
            // cursor mapped through the wrong origin still lands on a frame
            // at (0, 0), and would go unnoticed here.
            div().size_full().p(INSET).child(
                FlameGraph::new(tree())
                    .id("flame")
                    .children(|node: &Node| node.children.as_slice())
                    .value(|node: &Node| node.value)
                    .label(|node: &Node| node.label.clone())
                    .on_click(move |path, _, _| clicked.borrow_mut().push(path.clone())),
            )
        }
    }

    /// Draw the chart in a window and hand back the click log and the window's
    /// width, which the tests place their clicks against.
    fn draw(
        cx: &mut TestAppContext,
    ) -> (
        Rc<RefCell<Vec<FlamePath>>>,
        &mut gpui::VisualTestContext,
        f32,
    ) {
        cx.update(crate::init);
        let clicked = Rc::new(RefCell::new(vec![]));
        let (_, cx) = cx.add_window_view({
            let clicked = clicked.clone();
            |_, _| FlameView { clicked }
        });
        let width = cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            window.viewport_size().width.as_f32() - INSET.as_f32() * 2.
        });
        (clicked, cx, width)
    }

    /// A window point over the frame `fraction` across `width` in `row`.
    fn at(width: f32, fraction: f32, row: usize) -> Point<Pixels> {
        point(
            INSET + px(width * fraction),
            INSET + px(row as f32 * ROW_HEIGHT + 4.),
        )
    }

    #[gpui::test]
    fn the_cursor_reaches_a_plot_away_from_the_window_origin(cx: &mut TestAppContext) {
        // `Plot::tooltip_state` receives a cursor already relative to the
        // plot's origin. Subtracting the origin a second time only misses when
        // the plot is away from the window's corner, which is every real
        // layout and was not what the window tests below used to exercise.
        let mut chart = FlameGraph::new(tree())
            .children(|node: &Node| node.children.as_slice())
            .value(|node: &Node| node.value);
        chart.domain = Some(Domain {
            start: 0.,
            end: 100.,
        });
        let bounds = gpui::bounds(point(px(540.), px(420.)), gpui::size(px(200.), px(100.)));

        cx.update(|cx| {
            // A tenth of the way across the plot, on its first row.
            let state = chart.tooltip_state(point(px(20.), px(4.)), bounds, cx);
            assert!(state.is_some(), "a cursor inside the plot resolves a frame");

            // The window-absolute cursor is not a plot coordinate, and here it
            // lands far outside a 200x100 plot.
            let absolute = chart.tooltip_state(point(px(560.), px(424.)), bounds, cx);
            assert!(absolute.is_none(), "the plot is only 200x100");
        });
    }

    #[gpui::test]
    fn a_click_zooms_to_the_frame_under_the_cursor(cx: &mut TestAppContext) {
        let (clicked, cx, width) = draw(cx);

        // Row 0 at 10% of the width: the first root, which spans 0..60%.
        cx.simulate_click(at(width, 0.1, 0), Modifiers::default());
        // Row 0 past 60%: the second root.
        cx.simulate_click(at(width, 0.8, 0), Modifiers::default());
        // Row 1 at 10%: the first root's child, which spans 0..20%.
        cx.simulate_click(at(width, 0.1, 1), Modifiers::default());

        assert_eq!(
            *clicked.borrow(),
            vec![
                FlamePath::root(0),
                FlamePath::root(1),
                FlamePath::root(0).child(0),
            ]
        );
    }

    #[gpui::test]
    fn a_click_over_self_time_zooms_nothing(cx: &mut TestAppContext) {
        let (clicked, cx, width) = draw(cx);

        // Row 1 at 40%: inside "main" but past its only child, so the cursor is
        // over main's own time, which is not a frame.
        cx.simulate_click(at(width, 0.4, 1), Modifiers::default());

        assert!(clicked.borrow().is_empty());
    }

    #[gpui::test]
    fn hovering_a_frame_draws_a_tooltip_over_it(cx: &mut TestAppContext) {
        let (_, cx, width) = draw(cx);

        // The hover fades in over several frames, and the tooltip re-resolves
        // the frame under the cursor on each of them.
        cx.simulate_mouse_move(at(width, 0.1, 0), None, Modifiers::default());
        for _ in 0..3 {
            cx.update(|window, cx| window.draw(cx).clear(cx));
        }

        // The cursor leaves: the hover lingers over the last frame while it
        // fades back out, so the tooltip must still resolve it.
        cx.simulate_mouse_move(point(INSET, px(-20.)), None, Modifiers::default());
        for _ in 0..3 {
            cx.update(|window, cx| window.draw(cx).clear(cx));
        }
    }

    #[gpui::test]
    fn a_drag_is_not_a_click(cx: &mut TestAppContext) {
        let (clicked, cx, width) = draw(cx);

        // A press that travels further than the slop is someone scrolling the
        // stack, not choosing a frame.
        let start = at(width, 0.1, 0);
        cx.simulate_mouse_down(start, MouseButton::Left, Modifiers::default());
        cx.simulate_mouse_up(
            point(start.x, start.y + px(40.)),
            MouseButton::Left,
            Modifiers::default(),
        );

        assert!(clicked.borrow().is_empty());
    }

    #[gpui::test]
    fn a_release_without_a_press_on_the_chart_zooms_nothing(cx: &mut TestAppContext) {
        let (clicked, cx, width) = draw(cx);

        // A drag that began elsewhere and happens to end over the chart.
        cx.simulate_mouse_up(at(width, 0.1, 0), MouseButton::Left, Modifiers::default());

        assert!(clicked.borrow().is_empty());
    }
}
