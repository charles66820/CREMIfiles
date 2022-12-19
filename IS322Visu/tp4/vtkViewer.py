import vtk

dataPath = "data/"

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
  # headScalarComponent = readerData.GetPointData().GetScalars("head")
  # print(headScalarComponent.GetRange())

  print("Head scalars value range :")
  headScalarRange = readerData.GetScalarRange()
  print(headScalarRange)

  # filters

  ## iso-surface
  contour = vtk.vtkContourFilter()
  contour.SetInputData(reader.GetOutput())
  contour.SetValue(0, 20)

  # mapper
  mapper = vtk.vtkDataSetMapper()
  # mapper.ScalarVisibilityOff()
  mapper.SetScalarRange(headScalarRange)
  mapper.SetInputConnection(contour.GetOutputPort())
  return (mapper, contour)

def addScalarBarWidget(interactor, objMapper):
  lut = vtk.vtkLookupTable()
  lut.Build()
  objMapper.SetLookupTable(lut)

  scalarBar = vtk.vtkScalarBarActor()
  scalarBar.SetOrientationToHorizontal()
  scalarBar.SetLookupTable(lut)
  scalarBar.SetTitle("Density range")

  scalarBarWidget = vtk.vtkScalarBarWidget()
  scalarBarWidget.SetInteractor(interactor)
  scalarBarWidget.SetScalarBarActor(scalarBar)
  scalarBarWidget.EnabledOn()
  return scalarBarWidget

class SliderObserver(object):
  def __init__(self, slider, contour):
    self.slider = slider
    self.contour = contour

  def __call__(self, caller, ev):
    self.contour.SetValue(0, self.slider.GetValue())

def addSliderWidget(interactor, objMapper, contour):
  slider = vtk.vtkSliderRepresentation2D()

  objScalarRange = objMapper.GetScalarRange()
  slider.SetMinimumValue(objScalarRange[0])
  slider.SetMaximumValue(objScalarRange[1])
  slider.SetValue(contour.GetValue(0))
  slider.SetTitleText("Contour value")

  slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
  slider.GetPoint1Coordinate().SetValue(0.1, 0.1)
  slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
  slider.GetPoint2Coordinate().SetValue(0.4, 0.1)

  slider.SetTubeWidth(0.008)
  slider.SetSliderLength(0.008)
  slider.SetSliderWidth(0.08)
  slider.SetTitleHeight(0.04)
  slider.SetLabelHeight(0.04)

  sliderWidget = vtk.vtkSliderWidget()
  sliderWidget.SetInteractor(interactor)
  sliderWidget.SetRepresentation(slider)
  sliderWidget.SetAnimationModeToAnimate()
  sliderWidget.EnabledOn()

  sliderWidget.AddObserver("InteractionEvent", SliderObserver(slider, contour))
  return sliderWidget

def addConeActor(renderer, interactor):
  coneActor = vtk.vtkActor()
  coneActor.SetMapper(getConeMapper())
  renderer.AddActor(coneActor)
  return coneActor

def addHeadActor(renderer, interactor):
  headActor = vtk.vtkActor()
  (headMapper, contour) = getHeadMapper()
  headActor.SetMapper(headMapper)
  # show the edges of the image grid
  # headActor.GetProperty().SetRepresentationToWireframe()
  renderer.AddActor(headActor)

  # headMapper = headActor.GetMapper()
  scalarBarWidget = addScalarBarWidget(interactor, headMapper)
  sliderWidget = addSliderWidget(interactor, headMapper, contour)

  return (scalarBarWidget, sliderWidget)

def createRenderer():
  renderer = vtk.vtkRenderer()
  renderer.SetBackground(0.0, 0.0, 0.0)

  camera = renderer.GetActiveCamera()
  camera.SetRoll(180)
  camera.Elevation(90)
  camera.Azimuth(0)
  return renderer

def createWindow():
  renderer = createRenderer()

  window = vtk.vtkRenderWindow()
  window.SetSize(500, 500)
  window.SetWindowName("VTK Viewer")
  window.AddRenderer(renderer)

  interactor = vtk.vtkRenderWindowInteractor()
  interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
  interactor.SetRenderWindow(window)

  # addConeActor(renderer, interactor)
  _ = addHeadActor(renderer, interactor)

  # Always after add actors to render
  renderer.ResetCamera()

  window.Render()
  interactor.Start()

def main():
  createWindow()

if __name__ == "__main__":
  main()
