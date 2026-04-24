import open3d as o3d
from open3d.visualization import rendering
import torch
import numpy as np
import imageio
from tqdm import tqdm

w, h = 640, 480
aspect = h/w
s = 2
render = None

def init_renderer():
    global render
    if render is None:
        render = rendering.OffscreenRenderer(w, h)
        render.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
        render.scene.scene.enable_sun_light(True)
        render.scene.show_skybox(False)

def get_video():
    import os
    
    os.makedirs('./tmp', exist_ok=True)
    _, _, files = next(os.walk("tmp/"))
    file_count = len(files)
    return "tmp/output" + str(file_count) + ".mp4"

def render_mesh(verts, face):
    global w, h, aspect, s, render
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(verts[0].numpy()[:,:])
    mesh.triangles=o3d.utility.Vector3iVector(face)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.5, 0.5, 0.5))
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
    render.scene.remove_geometry("box1")
    images = []
    for i in tqdm(range(1,len(verts))):
        render.scene.add_geometry("box", mesh, material)
        images.append(render.render_to_image())
        render.scene.remove_geometry("box")
        mesh.vertices=o3d.utility.Vector3dVector(verts[i].numpy()[:,:])
        
    return images

def render_mesh_image_error(vert, face, error_faces):
    global w, h, aspect, s, render
    
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions=o3d.core.Tensor(vert[:,:].numpy(), o3d.core.float32)
    mesh.triangle.indices=o3d.core.Tensor(face.numpy(),o3d.core.int32)
    mesh.compute_vertex_normals()
    mesh.triangle.colors = o3d.core.Tensor(error_faces.numpy(), o3d.core.float32)
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.remove_geometry("box")
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
    render.scene.add_geometry("box1", mesh, material)
        
    image = render.render_to_image()
    render.scene.remove_geometry("box1")
        
    return image

def render_error_mesh(verts, face, error_faces):
    global w, h, aspect, s, render
    
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions=o3d.core.Tensor(verts[0,:,:].numpy(), o3d.core.float32)
    mesh.triangle.indices=o3d.core.Tensor(face.numpy(),o3d.core.int32)
    mesh.compute_vertex_normals()
    mesh.triangle.colors = o3d.core.Tensor(error_faces[0].numpy(), o3d.core.float32)
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
    render.scene.remove_geometry("box1")
    images = []
    for i in tqdm(range(1,len(verts))):
        render.scene.add_geometry("box", mesh, material)
        images.append(render.render_to_image())
        render.scene.remove_geometry("box")
        mesh.vertex.positions=o3d.core.Tensor(verts[i,:,:].numpy(), o3d.core.float32)
        mesh.triangle.colors = o3d.core.Tensor(error_faces[i].numpy(), o3d.core.float32)

    return images

def render_error_mesh_side(verts1, verts2, face, error_faces1, error_faces2):
    global w, h, aspect, s, render
    # here!!!
    
    mesh1 = o3d.t.geometry.TriangleMesh()
    mesh1.vertex.positions=o3d.core.Tensor(verts1[0,:,:].numpy(), o3d.core.float32)
    mesh1.triangle.indices=o3d.core.Tensor(face.numpy(),o3d.core.int32)
    mesh1.compute_vertex_normals()
    mesh1.triangle.colors = o3d.core.Tensor(error_faces1[0].numpy(), o3d.core.float32)
    
    mesh2 = o3d.t.geometry.TriangleMesh()
    mesh2.vertex.positions=o3d.core.Tensor(verts2[0,:,:].numpy(), o3d.core.float32)
    mesh2.triangle.indices=o3d.core.Tensor(face.numpy(),o3d.core.int32)
    mesh2.compute_vertex_normals()
    mesh2.triangle.colors = o3d.core.Tensor(error_faces2[0].numpy(), o3d.core.float32)
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
    render.scene.remove_geometry("box1")
    images = []
    for i in tqdm(range(1,len(verts1))):
        render.scene.add_geometry("box1", mesh1, material)
        render.scene.add_geometry("box2", mesh2, material)
        images.append(render.render_to_image())
        render.scene.remove_geometry("box1")
        render.scene.remove_geometry("box2")
        
        mesh1.vertex.positions=o3d.core.Tensor(verts1[i,:,:].numpy(), o3d.core.float32)
        mesh1.triangle.colors = o3d.core.Tensor(error_faces1[i].numpy(), o3d.core.float32)
        
        mesh2.vertex.positions=o3d.core.Tensor(verts2[i,:,:].numpy(), o3d.core.float32)
        mesh2.triangle.colors = o3d.core.Tensor(error_faces2[i].numpy(), o3d.core.float32)

    return images

def render_mesh_multi(verts, face):
    global w, h, aspect, s, render
    
    start_tran = np.array([0.,0,1.5])
    end_tran = np.array([0.,0,-2])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(verts[0].numpy()[0,:,:]+start_tran)
    mesh.triangles=o3d.utility.Vector3iVector(face)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0., 0.5, 0.5))
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 0], [50, 0, 0], [0, 1, 0])

    
    for i in tqdm(range(1,len(verts))):
        render.scene.add_geometry("box"+str(i), mesh, material)
        cur_tran = start_tran * ((len(verts)-i)/float(len(verts))) + end_tran * ((i)/float(len(verts)))
        mesh.vertices=o3d.utility.Vector3dVector(verts[i].numpy()[0,:,:] + cur_tran)
        
    image = render.render_to_image()

    for i in range(1,len(verts)):
        render.scene.remove_geometry("box"+str(i))
        
    return image

def render_mesh_image(vert, face):
    global w, h, aspect, s, render
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vert.numpy()[0,:,:])
    mesh.triangles=o3d.utility.Vector3iVector(face)
    mesh.compute_vertex_normals()
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color((0., 0., 0.))
    
    material = rendering.MaterialRecord()
    material.shader = 'unlitLine'
    material.line_width = 3
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
    render.scene.add_geometry("box1", wireframe, material)
        
    image = render.render_to_image()
    render.scene.remove_geometry("box1")
        
    return image

def render_mesh_image_error_multi(verts, face, error_faces):
    global w, h, aspect, s, render
    
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions=o3d.core.Tensor(verts[0,:,:].numpy(), o3d.core.float32)
    mesh.triangle.indices=o3d.core.Tensor(face.numpy(),o3d.core.int32)
    mesh.compute_vertex_normals()
    mesh.triangle.colors = o3d.core.Tensor(error_faces[0].numpy(), o3d.core.float32)
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    for i in range(0,verts.shape[0]-1):
        render.scene.add_geometry("box"+str(i), mesh, material)
        mesh.vertex.positions=o3d.core.Tensor(verts[i+1,:,:].numpy(), o3d.core.float32)
        mesh.triangle.colors = o3d.core.Tensor(error_faces[i+1].numpy(), o3d.core.float32)

    render.scene.add_geometry("box"+str(verts.shape[0]-1), mesh, material)
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    # render.scene.camera.look_at([0, 0, 1], [0, 50, 0], [0, 0, 1]) # Seq 25
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
        
    image = render.render_to_image()
    render.scene.remove_geometry("box")
    for i in range(0,10):
        render.scene.remove_geometry("box"+str(i))
        
    return image

def render_mesh_image_error(vert, face, error_faces):
    global w, h, aspect, s, render
    
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions=o3d.core.Tensor(vert[:,:].numpy(), o3d.core.float32)
    mesh.triangle.indices=o3d.core.Tensor(face.numpy(),o3d.core.int32)
    mesh.compute_vertex_normals()
    mesh.triangle.colors = o3d.core.Tensor(error_faces.numpy(), o3d.core.float32)
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.remove_geometry("box")
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 1], [50, 0, 0], [0, 0, 1])
    render.scene.add_geometry("box1", mesh, material)
        
    image = render.render_to_image()
    render.scene.remove_geometry("box1")
        
    return image

def save_frames(images):
    global w, h, aspect, s, render
    # Ensure images are in uint8 format (0-255)
    frames = [(255-img*255).astype('uint8') for img in images]
    video_path = get_video()
    
    # Write to video file (adjust fps as needed)
    with imageio.get_writer(video_path, fps=60) as writer:
        for img in frames:
            writer.append_data(img)

    return video_path
    
def save_frames_color(images):
    # Ensure images are in uint8 format (0-255)
    frames = [img.astype('uint8') for img in images]
    video_path = get_video()
    
    # Write to video file (adjust fps as needed)
    with imageio.get_writer(video_path, fps=60) as writer:
        for img in frames:
            writer.append_data(img)

    return video_path
            
def render_skeleton_points(joints):
    global w, h, aspect, s, render
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(joints[0,:,:].numpy())
    points.paint_uniform_color((0.5, 0., 0.))
    
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit'
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 0], [50, 0, 0], [0, 0, 1])
    
    images = []
    for i in tqdm(range(1,len(joints))):
        render.scene.add_geometry("joints", points, material)
        images.append(render.render_to_image())
        render.scene.remove_geometry("joints")
        points.points=o3d.utility.Vector3dVector(joints[i,:,:].numpy())
        
    return images

def render_skeleton(joints, bones):
    global w, h, aspect, s, render
    points = o3d.geometry.LineSet()
    points.points = o3d.utility.Vector3dVector(joints[0,:,:].numpy())
    points.lines = o3d.utility.Vector2iVector(bones)
    points.paint_uniform_color((0.5, 0., 0.))
    
    material = rendering.MaterialRecord()
    material.shader = 'unlitLine'
    material.line_width = 5
    
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,-s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 0], [50, 0, 0], [0, 0, 1])
    
    images = []
    for i in tqdm(range(1,len(joints))):
        render.scene.add_geometry("bones", points, material)
        images.append(render.render_to_image())
        render.scene.remove_geometry("bones")
        points.points=o3d.utility.Vector3dVector(joints[i,:,:].numpy())
        
    return images