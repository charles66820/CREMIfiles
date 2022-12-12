import vtk

dataPath = "data/"

def createWindow(renderer):
  window = vtk.vtkRenderWindow()
  window.SetSize(500, 500)
  window.SetWindowName("VTK Viewer")
  window.AddRenderer(renderer)

  interactor = vtk.vtkRenderWindowInteractor()
  interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
  interactor.SetRenderWindow(window)

  window.Render()
  interactor.Start()

def coneMapper():
  cone = vtk.vtkConeSource()

  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(cone.GetOutputPort())
  return mapper

def headMapper():
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName(dataPath + "head.vti")
  reader.Update()

  readerData = reader.GetOutput()
  headComponent = readerData.GetPointData().GetScalars("head")

  print("Head scalars value range :")
  print(headComponent.GetRange())

  # filters
  vtk.vtkContourFilter()

  ## iso-surface
  contour = vtk.vtkContourFilter()
  contour.SetInputData(reader.GetOutput())
  contour.SetValue(0, 20)

  # mapper
  mapper = vtk.vtkDataSetMapper()
  # mapper.ScalarVisibilityOff()
  mapper.SetInputConnection(contour.GetOutputPort())
  return mapper

def main():
  actor = vtk.vtkActor()
#   actor.SetMapper(coneMapper())
  actor.SetMapper(headMapper())
  # show the edges of the image grid
#   actor.GetProperty().SetRepresentationToWireframe()

  renderer = vtk.vtkRenderer()
  renderer.SetBackground(0.0, 0.0, 0.0)
  renderer.AddActor(actor)

  camera = renderer.GetActiveCamera()
  camera.SetRoll(180);
  camera.Elevation(90);
  camera.Azimuth(0);
  renderer.ResetCamera();

  createWindow(renderer)

if __name__ == "__main__":
  main()
