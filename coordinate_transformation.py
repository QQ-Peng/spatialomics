# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# %%
def calculate_affine_transform(A_coords, B_coords):
    """
    根据三组对应点计算从坐标系A到坐标系B的仿射变换矩阵
    """
    # 构造齐次坐标矩阵
    P = np.array([
        [A_coords[0][0], A_coords[1][0], A_coords[2][0]],
        [A_coords[0][1], A_coords[1][1], A_coords[2][1]],
        [1, 1, 1]
    ])
    
    Q = np.array([
        [B_coords[0][0], B_coords[1][0], B_coords[2][0]],
        [B_coords[0][1], B_coords[1][1], B_coords[2][1]],
        [1, 1, 1]
    ])
    
    try:
        P_inv = np.linalg.inv(P)
        T = Q @ P_inv
        return T
    except np.linalg.LinAlgError:
        raise ValueError("给定的三个点共线，无法计算唯一的仿射变换矩阵。")

def transform_point(T, point):
    """
    使用仿射变换矩阵将点从坐标系A转换到坐标系B
    """
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_homogeneous = T @ point_homogeneous
    return (transformed_homogeneous[0], transformed_homogeneous[1])

def decompose_affine_matrix(T):
    """
    分解仿射变换矩阵为平移、旋转、缩放分量
    """
    # 提取线性变换部分 (左上角2x2矩阵)
    linear_part = T[:2, :2]
    
    # 提取平移部分 (右上角2x1向量)
    translation = T[:2, 2]
    
    # 使用奇异值分解(SVD)来分解线性部分
    U, S, Vt = np.linalg.svd(linear_part)
    
    # 旋转矩阵 (R = U × Vt)
    rotation = U @ Vt
    
    # 缩放矩阵 (对角线元素)
    scale = S
    
    # 计算旋转角度 (弧度)
    angle_rad = np.arctan2(rotation[1, 0], rotation[0, 0])
    angle_deg = np.degrees(angle_rad)
    
    return translation, rotation, scale, angle_deg

def plot_coordinate_systems(A_coords, B_coords, d_point_A, d_point_B, T):
    """
    绘制两个坐标系和点的位置
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制坐标系A
    ax1.scatter([p[0] for p in A_coords], [p[1] for p in A_coords], c='red', s=100, label='Reference Points')
    ax1.scatter(d_point_A[0], d_point_A[1], c='blue', s=150, marker='*', label='Point d')
    for i, (x, y) in enumerate(A_coords):
        ax1.text(x + 0.1, y + 0.1, f'a,b,c[{i}]', fontsize=12)
    ax1.text(d_point_A[0] + 0.1, d_point_A[1] + 0.1, 'd', fontsize=12)
    ax1.set_title('Coordinate System A')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # 绘制坐标系B
    ax2.scatter([p[0] for p in B_coords], [p[1] for p in B_coords], c='red', s=100, label='Reference Points')
    ax2.scatter(d_point_B[0], d_point_B[1], c='blue', s=150, marker='*', label='Point d')
    for i, (x, y) in enumerate(B_coords):
        ax2.text(x + 0.1, y + 0.1, f'a,b,c[{i}]', fontsize=12)
    ax2.text(d_point_B[0] + 0.1, d_point_B[1] + 0.1, "d'", fontsize=12)
    ax2.set_title('Coordinate System B')
    ax2.set_xlabel("X'")
    ax2.set_ylabel("Y'")
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

# %%
# 定义包含平移、旋转、缩放的变换示例
def main():
    print("=== 仿射变换示例：平移 + 旋转 + 缩放 ===\n")
    
    # 在坐标系A中的三个点（构成一个三角形）
    A_coords = [(1, 1), (4, 1), (2.5, 4)]
    
    # 定义变换参数
    translation = (3, 2)        # 平移：向右3个单位，向上2个单位
    rotation_angle = 30         # 旋转：30度（逆时针）
    scale_factor = (1.5, 1.5)   # 缩放：x方向1.5倍，y方向1.5倍
    
    print(f"应用的变换参数:")
    print(f"  平移: {translation}")
    print(f"  旋转: {rotation_angle}°")
    print(f"  缩放: {scale_factor}")
    print()
    
    # 手动计算变换后的B坐标（用于验证）
    def apply_transform(point):
        # 先缩放
        scaled = (point[0] * scale_factor[0], point[1] * scale_factor[1])
        # 再旋转（角度转换为弧度）
        angle_rad = np.radians(rotation_angle)
        rotated = (
            scaled[0] * np.cos(angle_rad) - scaled[1] * np.sin(angle_rad),
            scaled[0] * np.sin(angle_rad) + scaled[1] * np.cos(angle_rad)
        )
        # 最后平移
        translated = (rotated[0] + translation[0], rotated[1] + translation[1])
        return translated
    
    # 计算三个点在坐标系B中的坐标
    B_coords = [apply_transform(point) for point in A_coords]
    
    print("三个参考点的坐标:")
    for i, (a_point, b_point) in enumerate(zip(A_coords, B_coords)):
        print(f"  点{i+1}: A{a_point} → B({b_point[0]:.2f}, {b_point[1]:.2f})")
    
    # 要转换的点d在坐标系A中的坐标
    d_point_A = (2.5, 2.5)
    d_point_B_expected = apply_transform(d_point_A)
    
    print(f"\n要转换的点d: A{d_point_A}")
    print(f"预期在B中的坐标: ({d_point_B_expected[0]:.2f}, {d_point_B_expected[1]:.2f})")
    print()
    
    try:
        # 1. 计算仿射变换矩阵
        T_matrix = calculate_affine_transform(A_coords, B_coords)
        print("计算得到的仿射变换矩阵 T:")
        print(np.round(T_matrix, 4))
        print()
        
        # 2. 分解变换矩阵
        trans, rot, scale, angle = decompose_affine_matrix(T_matrix)
        print("分解变换矩阵:")
        print(f"  平移分量: [{trans[0]:.4f}, {trans[1]:.4f}]")
        print(f"  旋转角度: {angle:.2f}°")
        print(f"  缩放因子: [{scale[0]:.4f}, {scale[1]:.4f}]")
        print(f"  旋转矩阵:\n{np.round(rot, 4)}")
        print()
        
        # 3. 使用变换矩阵转换点d
        d_point_B_calculated = transform_point(T_matrix, d_point_A)
        print(f"点d在坐标系B中的计算坐标: ({d_point_B_calculated[0]:.4f}, {d_point_B_calculated[1]:.4f})")
        print(f"与预期值的误差: {np.linalg.norm(np.array(d_point_B_calculated) - np.array(d_point_B_expected)):.6f}")
        print()
        
        # 4. 验证所有点的变换
        print("验证所有参考点的变换:")
        for i, point_A in enumerate(A_coords):
            point_B_calc = transform_point(T_matrix, point_A)
            error = np.linalg.norm(np.array(point_B_calc) - np.array(B_coords[i]))
            print(f"  点{i+1}: 计算误差 = {error:.8f}")
        
        # 5. 绘制坐标系
        plot_coordinate_systems(A_coords, B_coords, d_point_A, d_point_B_calculated, T_matrix)
        
    except ValueError as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
# %%
