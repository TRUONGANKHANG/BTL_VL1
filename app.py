"""Interactive PH1003 Biot-Savart circular loop simulation."""
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from magnetic_field import field_at_points, axis_field, center_field

st.set_page_config(page_title="Biot-Savart | PH1003", page_icon="🧲", layout="wide")
st.title("🧲 Từ trường của dòng điện tròn")
st.caption("BTL Vật lý 1 (PH1003) · Định luật Biot–Savart · Phương pháp điểm giữa")
st.markdown("Mô hình: vòng dây mảnh nằm trong mặt phẳng **Oxy**, tâm tại **O**. Dòng điện **I > 0** chạy ngược chiều kim đồng hồ khi nhìn từ **+Oz**; từ trường tại tâm hướng **+Oz**.")

with st.sidebar:
    st.header("Thông số mô phỏng")
    R = st.slider("Bán kính R (m)", 0.02, 0.30, 0.10, 0.01)
    I = st.slider("Dòng điện I (A)", -10.0, 10.0, 5.0, 0.5)
    N = st.slider("Số phần tử N", 20, 1000, 200, 20)
    density = st.slider("Độ phân giải lưới", 17, 61, 31, 2)
    st.info("Điểm nằm ngay trên dây dẫn lý tưởng không có giá trị từ trường hữu hạn trong mô hình này. Đồ thị loại các điểm đó.")

st.latex(r"\vec B(M) \approx \frac{\mu_0 I}{4\pi}\sum_{i=0}^{N-1}\frac{\Delta\vec\ell_i\times\vec r_i}{|\vec r_i|^3}")

# Reuse the same data array for all plot types.
@st.cache_data(show_spinner=False)
def compute_grid(r, current, n, count):
    x = np.linspace(-2.5*r, 2.5*r, count)
    z = np.linspace(-2.5*r, 2.5*r, count)
    X, Z = np.meshgrid(x, z)
    pts = np.stack((X, np.zeros_like(X), Z), axis=-1)
    B = field_at_points(pts, r, current, n)
    return x, z, X, Z, B

x, z, X, Z, field = compute_grid(R, I, N, density)
Bx, Bz = field[..., 0], field[..., 2]
Bm = np.linalg.norm(field, axis=-1)
# Avoid singular-looking oversized glyphs near the ideal conductor.
finite = np.isfinite(Bm)
plot_norm = np.where(finite, Bm, np.nan)
upper = np.nanpercentile(plot_norm, 93) if np.any(finite) else 0
strong = finite & (Bm <= max(upper, 1e-30))

col1, col2, col3 = st.columns(3)
col1.metric("B tại tâm (lý thuyết)", f"{abs(center_field(R, I))*1e6:.3f} µT")
col2.metric("Số đoạn chia", f"{N:,}")
col3.metric("Số điểm lưới", f"{density*density:,}")

plot_tab, compare_tab, data_tab = st.tabs(["Trường vectơ & đường sức", "Kiểm chứng & hội tụ", "Điểm khảo sát & CSV"])

with plot_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Vectơ từ trường (mặt phẳng Oxz)")
        fig, ax = plt.subplots(figsize=(7, 6))
        # Normalize display arrows; direction visualized, size not quantitative.
        v = np.sqrt(Bx*Bx + Bz*Bz)
        denom = np.where(v > 0, v, np.nan)
        u, w = Bx/denom, Bz/denom
        mask = np.isfinite(u) & np.isfinite(w) & strong
        ax.quiver(X[mask], Z[mask], u[mask], w[mask], angles="xy", scale_units="xy", scale=22/max(R, 1e-6), width=0.0025)
        ax.scatter([-R, R], [0, 0], s=38, c=["tab:blue", "tab:blue"], label="Giao điểm vòng dây")
        ax.set(xlabel="x (m)", ylabel="z (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(loc="upper right")
        st.pyplot(fig)
        st.caption("Các mũi tên được chuẩn hóa độ dài để nhìn rõ hướng; **không** thể hiện độ lớn B. Vùng có B quá lớn gần dây được ẩn bớt.")
        plt.close(fig)
    with right:
        st.subheader("Đường sức từ (mặt phẳng Oxz)")
        fig, ax = plt.subplots(figsize=(7, 6))
        # Streamplot can be erratic at singular points: mask samples
        bx_mask = np.ma.masked_invalid(np.where(strong, Bx, np.nan))
        bz_mask = np.ma.masked_invalid(np.where(strong, Bz, np.nan))
        if np.any(np.isfinite(Bx)) and abs(I) > 1e-15:
            try:
                ax.streamplot(x, z, bx_mask, bz_mask, density=1.3, linewidth=1.0, arrowsize=1.0, color="tab:teal")
            except (ValueError, TypeError):
                st.warning("Không đủ dữ liệu hợp lệ để vẽ đường sức với lưới hiện tại. Hãy thử tăng độ phân giải.")
        else:
            st.info("I = 0 nên không có từ trường để vẽ đường sức.")
        ax.scatter([-R, R], [0, 0], s=38, c="tab:blue")
        ax.set(xlabel="x (m)", ylabel="z (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        st.pyplot(fig)
        st.caption("Các đường được tính số trong mặt phẳng đối xứng Oxz; không biểu diễn trường 3D đầy đủ.")
        plt.close(fig)

with compare_tab:
    st.subheader("Công thức lý thuyết trên trục Oz")
    st.latex(r"B_z(z)=\frac{\mu_0 I R^2}{2(R^2+z^2)^{3/2}}")
    z_axis = np.linspace(0, 2.0*R, 101)
    P = np.column_stack((np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis))
    numerical = field_at_points(P, R, I, N)[:, 2]
    theoretical = axis_field(z_axis, R, I)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_axis, theoretical*1e6, label="Giải tích", linewidth=2)
    ax.plot(z_axis, numerical*1e6, linestyle="--", label=f"Tính số (N={N})")
    ax.set(xlabel="z (m)", ylabel="Bz (µT)")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)
    st.caption("Trên trục Oz, quy tắc điểm giữa cho kết quả trùng công thức lý thuyết đến sai số làm tròn; nên đánh giá hội tụ tại điểm ngoài trục.")
    st.subheader("Hội tụ tại điểm ngoài trục M = (R/2, 0, R/2)")
    p = np.array([R/2, 0, R/2])
    n_values = [10, 20, 40, 80, 160, 320, 640, 1000]
    ref = field_at_points(p, R, I, 20000)
    ref_mag = np.linalg.norm(ref)
    records = []
    for n in n_values:
        b = field_at_points(p, R, I, n)
        delta = np.linalg.norm(b-ref)
        records.append({"N": n, "|B| (µT)": np.linalg.norm(b)*1e6, "Sai khác tương đối (%)": delta/ref_mag*100 if ref_mag>0 else np.nan})
    frame = pd.DataFrame(records)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(frame["N"], np.maximum(frame["Sai khác tương đối (%)"], 1e-14), marker="o")
    ax.set(xlabel="Số đoạn chia N", ylabel="Sai khác so với N=20000 (%)")
    ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)
    st.caption("Mốc N = 20000 là nghiệm **tham chiếu tính số**, không phải nghiệm giải tích chính xác. I = 0 thì không xác định được sai khác tương đối.")

with data_tab:
    st.subheader("Tính tại một điểm bất kỳ")
    a, b, c = st.columns(3)
    xx = a.number_input("x (m)", value=0.0, step=0.01, format="%.4f")
    yy = b.number_input("y (m)", value=0.0, step=0.01, format="%.4f")
    zz = c.number_input("z (m)", value=0.0, step=0.01, format="%.4f")
    point_B = field_at_points(np.array([xx, yy, zz]), R, I, N)
    if np.all(np.isfinite(point_B)):
        st.write(f"**Bx:** {point_B[0]*1e6:.6f} µT · **By:** {point_B[1]*1e6:.6f} µT · **Bz:** {point_B[2]*1e6:.6f} µT · **|B|:** {np.linalg.norm(point_B)*1e6:.6f} µT")
    else:
        st.warning("Điểm khảo sát nằm trên vòng dây lý tưởng, nên từ trường không xác định trong mô hình này.")
    st.subheader("Tải bảng số liệu lưới Oxz")
    table = pd.DataFrame({"x_m":X.ravel(), "y_m":np.zeros(X.size), "z_m":Z.ravel(), "Bx_T":field[...,0].ravel(), "By_T":field[...,1].ravel(), "Bz_T":field[...,2].ravel(), "B_T":Bm.ravel()})
    st.dataframe(table.head(20), hide_index=True, use_container_width=True)
    st.download_button("⬇️ Tải CSV", table.to_csv(index=False).encode("utf-8-sig"), file_name="biot_savart_field.csv", mime="text/csv")

st.divider()
st.caption("Mô phỏng mang tính học thuật: dòng điện không đổi, vòng dây mảnh lý tưởng, môi trường chân không, μ₀ ≈ 4π·10⁻⁷ H/m.")
