import vtk

dataPath = "data/"

def createWindow(renderer, lut):
  window = vtk.vtkRenderWindow()
  window.SetSize(500, 500)
  window.SetWindowName("VTK Viewer")
  window.AddRenderer(renderer)

  interactor = vtk.vtkRenderWindowInteractor()
  interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
  interactor.SetRenderWindow(window)

  scalarBar = vtk.vtkScalarBarActor()
  scalarBar.SetOrientationToHorizontal()
  scalarBar.SetLookupTable(lut)

  scalarBarWidget = vtk.vtkScalarBarWidget()
  scalarBarWidget.SetInteractor(interactor)
  scalarBarWidget.SetScalarBarActor(scalarBar)
  scalarBarWidget.On()

  window.Render()
  interactor.Start()

def getConeMapper():
  cone = vtk.vtkConeSource()

  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(cone.GetOutputPort())
  return mapper

def getHeadMapper():
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName(dataPath + "head.vti")
  reader.Update()

  readerData = reader.GetOutput()
  headScalarComponent = readerData.GetPointData().GetScalars("head")

  print("Head scalars value range :")
  print(headScalarComponent.GetRange())
  headScalarRange = readerData.GetScalarRange()
  print(headScalarRange)

  # filters
  vtk.vtkContourFilter()

  ## iso-surface
  contour = vtk.vtkContourFilter()
  contour.SetInputData(reader.GetOutput())
  contour.SetValue(0, 20)

  # mapper
  mapper = vtk.vtkDataSetMapper()
  # mapper.ScalarVisibilityOff()
  mapper.SetScalarRange(headScalarRange)
  mapper.SetInputConnection(contour.GetOutputPort())
  return mapper

def main():
  # cone
  coneActor = vtk.vtkActor()
  coneActor.SetMapper(getConeMapper())

  # head
  headMapper = getHeadMapper()

  lut = vtk.vtkLookupTable()
  lut.Build()
  headMapper.SetLookupTable(lut)

  headActor = vtk.vtkActor()
  headActor.SetMapper(headMapper)
  # show the edges of the image grid
  # headActor.GetProperty().SetRepresentationToWireframe()

  # render
  renderer = vtk.vtkRenderer()
  renderer.SetBackground(0.0, 0.0, 0.0)
  renderer.AddActor(headActor)
  # renderer.AddActor(coneActor)

  camera = renderer.GetActiveCamera()
  camera.SetRoll(180)
  camera.Elevation(90)
  camera.Azimuth(0)
  renderer.ResetCamera()

  createWindow(renderer, lut)

if __name__ == "__main__":
  main()
