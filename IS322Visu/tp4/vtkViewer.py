import vtk

dataPath = "data/"

def addScalarBarWidget(interactor, lut):
  scalarBar = vtk.vtkScalarBarActor()
  scalarBar.SetOrientationToHorizontal()
  scalarBar.SetLookupTable(lut)

  scalarBarWidget = vtk.vtkScalarBarWidget()
  scalarBarWidget.SetInteractor(interactor)
  scalarBarWidget.SetScalarBarActor(scalarBar)
  scalarBarWidget.On()
  return scalarBarWidget


def addSliderWidget(interactor):
  slider = vtk.vtkSliderRepresentation2D();

  #TODO: min, max, initial
  slider.SetMinimumValue(0.0);
  slider.SetMaximumValue(181.6048126220703);
  # slider.SetValue(tessellate.GetChordError());
  slider.SetTitleText("Contour value");

  slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay();
  slider.GetPoint1Coordinate().SetValue(0.1, 0.1);
  slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay();
  slider.GetPoint2Coordinate().SetValue(0.4, 0.1);

  slider.SetTubeWidth(0.008)
  slider.SetSliderLength(0.008)
  slider.SetSliderWidth(0.08)
  slider.SetTitleHeight(0.04)
  slider.SetLabelHeight(0.04)

  sliderWidget = vtk.vtkSliderWidget()
  sliderWidget.SetInteractor(interactor);
  sliderWidget.SetRepresentation(slider);
  sliderWidget.SetAnimationModeToAnimate();
  sliderWidget.EnabledOn();

  # sliderWidget.AddObserver(vtkCommand::InteractionEvent, callbackChordError);
  #TODO:
  return sliderWidget

def createWindow(renderer):
  # addConeActor(renderer)
  lut = addHeadActor(renderer)

  camera = renderer.GetActiveCamera()
  camera.SetRoll(180)
  camera.Elevation(90)
  camera.Azimuth(0)
  renderer.ResetCamera()

  window = vtk.vtkRenderWindow()
  window.SetSize(500, 500)
  window.SetWindowName("VTK Viewer")
  window.AddRenderer(renderer)

  interactor = vtk.vtkRenderWindowInteractor()
  interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
  interactor.SetRenderWindow(window)

  scalarBarWidget = addScalarBarWidget(interactor, lut)

  sliderWidget = addSliderWidget(interactor)

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

def addConeActor(renderer):
  coneActor = vtk.vtkActor()
  coneActor.SetMapper(getConeMapper())
  renderer.AddActor(coneActor)

def addHeadActor(renderer):
  headMapper = getHeadMapper()

  lut = vtk.vtkLookupTable()
  lut.Build()
  headMapper.SetLookupTable(lut)

  headActor = vtk.vtkActor()
  headActor.SetMapper(headMapper)
  # show the edges of the image grid
  # headActor.GetProperty().SetRepresentationToWireframe()
  renderer.AddActor(headActor)
  return lut

def main():
  # render
  renderer = vtk.vtkRenderer()
  renderer.SetBackground(0.0, 0.0, 0.0)

  createWindow(renderer)

if __name__ == "__main__":
  main()
