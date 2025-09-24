import cv2 as cv
import numpy as np

def create_cube_vertices(a=1.0):
    """Cria os 8 vértices de um cubo centrado na origem."""
    vertices = np.array([
        [-a, -a, -a], [a, -a, -a], [a, a, -a], [-a, a, -a],  # Face frontal
        [-a, -a, a], [a, -a, a], [a, a, a], [-a, a, a]    # Face traseira
    ], dtype=np.float32)
    return vertices

def create_cube_edges():
    """Define as arestas do cubo."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Face frontal
        (4, 5), (5, 6), (6, 7), (7, 4),  # Face traseira
        (0, 4), (1, 5), (2, 6), (3, 7)   # Conectores
    ]
    return edges

def create_cube_faces():
    """Define as 6 faces do cubo."""
    faces = [
        (0, 1, 2, 3),  # Frente
        (4, 5, 6, 7),  # Trás
        (0, 4, 7, 3),  # Esquerda
        (1, 5, 6, 2),  # Direita
        (0, 1, 5, 4),  # Baixo
        (3, 2, 6, 7)   # Cima
    ]
    return faces

def project_cube(vertices, rvec, tvec, K, dist_coeffs):
    """Projeta os vértices 3D para 2D."""
    img_points, _ = cv.projectPoints(vertices, rvec, tvec, K, dist_coeffs)
    return img_points.reshape(-1, 2)

def draw_wireframe_cube(image, projected_points, edges, color=(0, 255, 0), thickness=2):
    """Desenha o cubo em wireframe."""
    for i, j in edges:
        pt1 = tuple(projected_points[i].astype(int))
        pt2 = tuple(projected_points[j].astype(int))
        cv.line(image, pt1, pt2, color, thickness)

def draw_filled_cube(image, vertices_3d, rvec, tvec, K, dist_coeffs, faces, colors):
    """Desenha o cubo preenchido com faces coloridas e arestas."""
    # Calcula a profundidade média de cada face para ordenação
    face_depths = []
    for face_indices in faces:
        face_vertices_3d = vertices_3d[list(face_indices)]
        # Transforma os vértices da face para o sistema de coordenadas da câmera
        # R * X + T
        rotated_points = np.dot(cv.Rodrigues(rvec)[0], face_vertices_3d.T).T + tvec.T
        # A profundidade é o componente Z no sistema de coordenadas da câmera
        avg_depth = np.mean(rotated_points[:, 2])
        face_depths.append(avg_depth)

    # Ordena as faces da mais distante para a mais próxima (maior profundidade para menor)
    ordered_faces = [face for _, face in sorted(zip(face_depths, faces), key=lambda pair: pair[0], reverse=True)]

    # Desenha as faces preenchidas
    for i, face_indices in enumerate(ordered_faces):
        projected_face_points = project_cube(vertices_3d[list(face_indices)], rvec, tvec, K, dist_coeffs)
        cv.fillConvexPoly(image, projected_face_points.astype(int), colors[i % len(colors)])

    # Desenha as arestas por cima para acabamento
    projected_points = project_cube(vertices_3d, rvec, tvec, K, dist_coeffs)
    edges = create_cube_edges()
    draw_wireframe_cube(image, projected_points, edges, color=(0, 0, 0), thickness=1)

def main():
    width, height = 640, 480
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(255) # Fundo branco

    # Parâmetros da câmera
    focal_length = 500
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32) # Sem distorção

    # Posição do cubo
    rvec = np.array([0.5, 0.5, 0.0], dtype=np.float32) # Rotação inicial
    tvec = np.array([0.0, 0.0, 5.0], dtype=np.float32) # Translação (cubo à frente da câmera)

    vertices_3d = create_cube_vertices(a=1.0)
    edges = create_cube_edges()
    faces = create_cube_faces()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), # Vermelho, Verde, Azul
        (255, 255, 0), (0, 255, 255), (255, 0, 255) # Amarelo, Ciano, Magenta
    ]

    # 1. Cubo Wireframe
    wireframe_image = image.copy()
    projected_points_wireframe = project_cube(vertices_3d, rvec, tvec, K, dist_coeffs)
    draw_wireframe_cube(wireframe_image, projected_points_wireframe, edges)
    cv.imwrite("cubo_wireframe.png", wireframe_image)

    # 2. Cubo Preenchido
    filled_image = image.copy()
    draw_filled_cube(filled_image, vertices_3d, rvec, tvec, K, dist_coeffs, faces, colors)
    cv.imwrite("cubo_preenchido.png", filled_image)

    # 3. Animação
    video_filename = "cubo_animado.mp4"
    fps = 30
    duration = 5 # segundos
    total_frames = fps * duration

    fourcc = cv.VideoWriter_fourcc(*"mp4v") # Codec para MP4
    out = cv.VideoWriter(video_filename, fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = image.copy()
        angle = 2 * np.pi * i / total_frames # Rotação completa em 5 segundos
        # Rotação em torno do eixo Y e X
        rvec_animation = np.array([angle, angle, 0.0], dtype=np.float32)

        draw_filled_cube(frame, vertices_3d, rvec_animation, tvec, K, dist_coeffs, faces, colors)
        out.write(frame)

    out.release()
    print(f"Vídeo salvo como {video_filename}")

if __name__ == "__main__":
    main()


